"""
Training loop for micro-scale proof-of-concept experiments.
Logs loss curve, memory consumption, generalization on withheld validation.
Compares ROGM vs Transformer baseline on identical data.
Fits in 2GB RAM, 2 CPU, no GPU.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import psutil
import math
import random

from src.dataset import generate_batch, VOCAB_SIZE, SyntheticDataLoader
from src.models import get_model

# Ensure reproducibility
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    # no cuda

set_seed(42)
# Limit threads for 2 CPU
torch.set_num_threads(2)
torch.set_num_interop_threads(2)

device = torch.device("cpu")

def get_memory_mb():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    return mem

def train_one_model(model_name, config, log_prefix=""):
    print(f"\n{'='*60}")
    print(f"TRAINING {model_name.upper()} - config {config}")
    print(f"{'='*60}")
    d_model = config.get("d_model", 64)
    n_layers = config.get("n_layers", 2)
    batch_size = config.get("batch_size", 32)
    seq_len = config.get("seq_len", 32)
    max_len = config.get("max_len", 64)
    lr = config.get("lr", 1e-3)
    epochs = config.get("epochs", 30)
    num_batches_per_epoch = config.get("num_batches_per_epoch", 200)
    weight_decay = config.get("weight_decay", 0.01)

    model = get_model(model_name, vocab_size=VOCAB_SIZE, d_model=d_model, n_layers=n_layers, max_len=max_len, dropout=0.1)
    model.to(device)
    print(f"Model {model_name} params: {model.count_params()}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion = nn.CrossEntropyLoss(reduction='none') # we will mask

    # For logging
    train_losses = []
    val_losses = []
    val_accs = []
    val_long_losses = []
    val_long_accs = []

    # Generate fixed validation sets (deterministic)
    # Fix seed for validation generation
    val_seed_state = random.getstate()
    torch_state = torch.get_rng_state()
    random.seed(1234)
    torch.manual_seed(1234)
    val_batch_inp, val_batch_tgt, val_batch_mask = generate_batch(500, seq_len=seq_len) # 500*32
    val_long_inp, val_long_tgt, val_long_mask = generate_batch(500, seq_len=64) # longer generalization
    # Per-task validation for breakdown?
    # Generate per-task val sets for analysis
    # parity only
    random.seed(1234)
    torch.manual_seed(1234)
    # We'll generate 200 of each task for breakdown evaluation after training
    # keep val sets as above random mix; breakdown will be generated later
    random.setstate(val_seed_state)
    torch.set_rng_state(torch_state)

    val_batch_inp = val_batch_inp.to(device)
    val_batch_tgt = val_batch_tgt.to(device)
    val_batch_mask = val_batch_mask.to(device)
    val_long_inp = val_long_inp.to(device)
    val_long_tgt = val_long_tgt.to(device)
    val_long_mask = val_long_mask.to(device)

    best_val_loss = float('inf')
    start_time = time.time()
    initial_mem = get_memory_mb()
    print(f"Initial memory: {initial_mem:.1f} MB")

    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_tokens = 0
        epoch_start = time.time()
        for batch_idx in range(num_batches_per_epoch):
            inp, tgt, mask = generate_batch(batch_size, seq_len=seq_len)
            inp = inp.to(device)
            tgt = tgt.to(device)
            mask = mask.to(device)
            # forward
            logits = model(inp) # B,L,V
            # compute masked loss
            # logits view B*L, V ; tgt view B*L
            loss_per_token = criterion(logits.view(-1, VOCAB_SIZE), tgt.view(-1)) # B*L
            mask_flat = mask.view(-1)
            # masked mean: sum(loss*mask)/sum(mask)
            # Note mask may be 0 for filler; need to avoid div 0 but sum>0 always
            loss = (loss_per_token * mask_flat).sum() / (mask_flat.sum() + 1e-8)
            optimizer.zero_grad()
            loss.backward()
            # grad clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss_sum += loss.item() * mask_flat.sum().item()
            epoch_tokens += mask_flat.sum().item()
        scheduler.step()
        avg_train_loss = epoch_loss_sum / max(epoch_tokens, 1)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            # val
            logits_val = model(val_batch_inp)
            loss_per_token_val = criterion(logits_val.view(-1, VOCAB_SIZE), val_batch_tgt.view(-1))
            mask_flat_val = val_batch_mask.view(-1)
            val_loss = (loss_per_token_val * mask_flat_val).sum() / (mask_flat_val.sum() + 1e-8)
            val_losses.append(val_loss.item())
            # accuracy masked
            pred_val = logits_val.argmax(dim=-1) # B,L
            correct = ((pred_val == val_batch_tgt) * val_batch_mask.bool()).sum().item()
            total = val_batch_mask.sum().item()
            val_acc = correct / max(total,1)
            val_accs.append(val_acc)

            # val long
            logits_long = model(val_long_inp)
            loss_per_token_long = criterion(logits_long.view(-1, VOCAB_SIZE), val_long_tgt.view(-1))
            mask_flat_long = val_long_mask.view(-1)
            val_long_loss = (loss_per_token_long * mask_flat_long).sum() / (mask_flat_long.sum() + 1e-8)
            val_long_losses.append(val_long_loss.item())
            pred_long = logits_long.argmax(dim=-1)
            correct_long = ((pred_long == val_long_tgt) * val_long_mask.bool()).sum().item()
            total_long = val_long_mask.sum().item()
            val_long_acc = correct_long / max(total_long,1)
            val_long_accs.append(val_long_acc)

        mem = get_memory_mb()
        epoch_time = time.time() - epoch_start
        lr_cur = optimizer.param_groups[0]['lr']
        print(f"[Epoch {epoch:02d}/{epochs}] train_loss={avg_train_loss:.4f} val_loss={val_loss.item():.4f} val_acc={val_acc:.3f} | long_val_loss={val_long_loss.item():.4f} long_acc={val_long_acc:.3f} | lr={lr_cur:.5f} mem={mem:.1f}MB time={epoch_time:.1f}s")
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            # save best?
            # torch.save(model.state_dict(), f"{log_prefix}_{model_name}_best.pt")
        # early logs?

    total_time = time.time() - start_time
    final_mem = get_memory_mb()
    print(f"Training done {model_name}: total_time {total_time:.1f}s final_mem {final_mem:.1f}MB peak?")

    # Detailed per-task evaluation on held-out sets
    model.eval()
    per_task_metrics = {}
    with torch.no_grad():
        # generate per-task val sets in same way: 400 samples each
        # parity
        random.seed(999)
        torch.manual_seed(999)
        p_inp, p_tgt, p_mask = generate_batch(200, seq_len=32, task_probs=(1.0,0.0,0.0))
        s_inp, s_tgt, s_mask = generate_batch(200, seq_len=32, task_probs=(0.0,1.0,0.0))
        m_inp, m_tgt, m_mask = generate_batch(200, seq_len=32, task_probs=(0.0,0.0,1.0))
        # also longer for generalization
        p_inp_long, p_tgt_long, p_mask_long = generate_batch(200, seq_len=64, task_probs=(1.0,0.0,0.0))
        s_inp_long, s_tgt_long, s_mask_long = generate_batch(200, seq_len=64, task_probs=(0.0,1.0,0.0))
        m_inp_long, m_tgt_long, m_mask_long = generate_batch(200, seq_len=64, task_probs=(0.0,0.0,1.0))

        def eval_set(inp, tgt, mask):
            inp = inp.to(device); tgt = tgt.to(device); mask = mask.to(device)
            logits = model(inp)
            loss_per = criterion(logits.view(-1, VOCAB_SIZE), tgt.view(-1))
            mask_f = mask.view(-1)
            loss = (loss_per * mask_f).sum() / (mask_f.sum() + 1e-8)
            pred = logits.argmax(dim=-1)
            acc = ((pred == tgt) * mask.bool()).sum().item() / max(mask.sum().item(),1)
            return loss.item(), acc

        per_task_metrics['parity'] = eval_set(p_inp, p_tgt, p_mask)
        per_task_metrics['s3'] = eval_set(s_inp, s_tgt, s_mask)
        per_task_metrics['mqar'] = eval_set(m_inp, m_tgt, m_mask)
        per_task_metrics['parity_long'] = eval_set(p_inp_long, p_tgt_long, p_mask_long)
        per_task_metrics['s3_long'] = eval_set(s_inp_long, s_tgt_long, s_mask_long)
        per_task_metrics['mqar_long'] = eval_set(m_inp_long, m_tgt_long, m_mask_long)

    print(f"Per-task metrics for {model_name}:")
    for k,v in per_task_metrics.items():
        print(f"  {k}: loss={v[0]:.4f} acc={v[1]:.3f}")

    return {
        "model_name": model_name,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "val_long_losses": val_long_losses,
        "val_long_accs": val_long_accs,
        "per_task": per_task_metrics,
        "params": model.count_params(),
        "total_time": total_time,
        "final_mem_mb": final_mem,
        "initial_mem_mb": initial_mem,
        "config": config,
    }

def run_comparison(config=None):
    if config is None:
        config = {
            "d_model": 64,
            "n_layers": 2,
            "batch_size": 32,
            "seq_len": 32,
            "max_len": 64,
            "lr": 1.5e-3,
            "epochs": 30,
            "num_batches_per_epoch": 200, # 200*32=6400 sequences per epoch => ~6k *30 = 192k sequences
            "weight_decay": 0.01,
        }
    print(f"Config: {config}")
    # Ensure deterministic comparison: set seed before each model training? But we want same data distribution; generate_batch is random each call, so both models see different samples but same distribution. That's okay for fairness but we could also fix sequence seed per batch by using same random state? Simpler: each model sees independently sampled data from same distribution - still fair expectation.
    # To make more fair, we set same seed and generate same number steps, though random sampling will differ but distribution same.
    results = {}
    # Train transformer first
    set_seed(42)
    results["transformer"] = train_one_model("transformer", config, log_prefix="exp")
    # Reset seed for ROGM but not necessary to be identical
    set_seed(42)
    results["rogm"] = train_one_model("rogm", config, log_prefix="exp")

    # Print comparison summary
    print("\n"+"="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    for model_name in ["transformer", "rogm"]:
        r = results[model_name]
        print(f"\n{model_name.upper()}:")
        print(f"  Params: {r['params']}")
        print(f"  Final Train Loss: {r['train_losses'][-1]:.4f}")
        print(f"  Final Val Loss: {r['val_losses'][-1]:.4f} acc {r['val_accs'][-1]:.3f}")
        print(f"  Final Long Val Loss: {r['val_long_losses'][-1]:.4f} acc {r['val_long_accs'][-1]:.3f}")
        print(f"  Per-task:")
        for k,(loss,acc) in r['per_task'].items():
            print(f"    {k:12s} loss {loss:.4f} acc {acc:.3f}")
        print(f"  Time {r['total_time']:.1f}s Mem {r['final_mem_mb']:.1f}MB")

    # Compute win rates
    # Superior reasoning generalization defined as better accuracy on withheld val (especially long and parity/s3)
    # We'll calculate differences
    t = results["transformer"]
    r = results["rogm"]
    print("\nDELTA (ROGM - Transformer)  Positive means ROGM better for acc, lower for loss:")
    print(f"  Val Acc delta: {r['val_accs'][-1] - t['val_accs'][-1]:+.3f}")
    print(f"  Long Val Acc delta: {r['val_long_accs'][-1] - t['val_long_accs'][-1]:+.3f}")
    print(f"  Val Loss delta: {r['val_losses'][-1] - t['val_losses'][-1]:+.4f} (negative is better)")
    for k in t['per_task']:
        delta_acc = r['per_task'][k][1] - t['per_task'][k][1]
        delta_loss = r['per_task'][k][0] - t['per_task'][k][0]
        print(f"  {k}: acc delta {delta_acc:+.3f} loss delta {delta_loss:+.4f}")

    # Autonomously evaluate success criteria
    # Success if ROGM better on at least 2 of 3 key reasoning metrics: parity, s3, parity_long (state-tracking) and also not worse on mqar retrieval
    # Let's define:
    reasoning_tasks = ['parity', 's3', 'parity_long', 's3_long']
    wins = sum(1 for k in reasoning_tasks if r['per_task'][k][1] > t['per_task'][k][1] + 0.02) # 2% margin
    mqar_ok = r['per_task']['mqar'][1] >= t['per_task']['mqar'][1] - 0.05 # not worse than 5% drop
    long_win = r['val_long_accs'][-1] > t['val_long_accs'][-1] + 0.02
    print(f"\nEVALUATION:")
    print(f"  Reasoning wins (out of 4): {wins}/4 (need >=2)")
    print(f"  MQAR not catastrophically worse: {mqar_ok}")
    print(f"  Long generalization win: {long_win}")
    success = (wins >=2 and mqar_ok)
    print(f"  => ROGM SUCCESS? {success}")
    if success:
        print("  >>> Novel architecture exhibits superior reasoning/generalization curves vs baseline micro-transformer. Ready to optimize further / harder tasks.")
    else:
        print("  >>> Novel architecture did not yet surpass baseline on reasoning. Identify bottleneck and iterate Phase 2.")

    # Save logs
    import json, datetime
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().isoformat()
    log_path = f"logs/comparison_{timestamp.replace(':','-')}.json"
    # Convert to serializable
    serializable = {}
    for k,v in results.items():
        serializable[k] = {
            "train_losses": v["train_losses"],
            "val_losses": v["val_losses"],
            "val_accs": v["val_accs"],
            "val_long_losses": v["val_long_losses"],
            "val_long_accs": v["val_long_accs"],
            "per_task": {tk: {"loss": loss, "acc": acc} for tk,(loss,acc) in v["per_task"].items()},
            "params": v["params"],
            "total_time": v["total_time"],
            "final_mem_mb": v["final_mem_mb"],
            "config": v["config"],
        }
    # Add delta summary
    serializable["_summary"] = {
        "wins": wins,
        "mqar_ok": bool(mqar_ok),
        "long_win": bool(long_win),
        "success": bool(success),
        "delta_val_acc": r['val_accs'][-1] - t['val_accs'][-1],
        "delta_long_acc": r['val_long_accs'][-1] - t['val_long_accs'][-1],
        "timestamp": timestamp,
    }
    with open(log_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Logs saved to {log_path}")
    # Also save latest to logs/latest.json for iteration tracking
    with open("logs/latest.json", "w") as f:
        json.dump(serializable, f, indent=2)
    return results

if __name__ == "__main__":
    # Test quick
    import psutil
    # quick smoke test with reduced epochs for debugging?
    # For full experiment run_comparison
    run_comparison()
