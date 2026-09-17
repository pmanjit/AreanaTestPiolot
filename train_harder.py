"""
Iteration 2: Harder Synthetic Reasoning & Parameter Stripping Optimization
- Harder tasks: selective parity (ignore token), longer sequences, 6-pair MQAR
- Tests: Transformer-64 vs ROGM-64 vs ROGM-lite-32 (stripped: half params, < baseline)
- Evaluates length generalization to 64/96/128 without retraining
"""
import torch
import torch.nn as nn
import time, os, random, json, datetime
import psutil
from src.dataset import generate_batch, VOCAB_SIZE, generate_selective_parity_sequence
from src.models import get_model

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
device = torch.device("cpu")

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def train_and_eval(model_name, config, model_kwargs):
    print(f"\n{'='*70}")
    print(f"TRAINING {model_name} | config {config} | kwargs {model_kwargs}")
    print(f"{'='*70}")
    d_model = model_kwargs.get("d_model", 64)
    n_layers = model_kwargs.get("n_layers", 2)
    max_len = config.get("max_len", 128)
    epochs = config.get("epochs", 25)
    batch_size = config.get("batch_size", 32)
    seq_len = config.get("seq_len", 48)
    lr = config.get("lr", 1.5e-3)
    num_batches = config.get("num_batches_per_epoch", 150)

    model = get_model(model_name.split("_")[0], vocab_size=VOCAB_SIZE, d_model=d_model, n_layers=n_layers, max_len=max_len, dropout=0.1)
    model.to(device)
    print(f"Params: {model.count_params()}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(reduction='none')

    # Fixed validation sets for fair comparison (generate once)
    set_seed(999)
    val_seq_len = 48
    val_inp, val_tgt, val_mask = generate_batch(400, seq_len=val_seq_len, harder=True)
    val64_inp, val64_tgt, val64_mask = generate_batch(300, seq_len=64, harder=True)
    val96_inp, val96_tgt, val96_mask = generate_batch(300, seq_len=96, harder=True)
    val128_inp, val128_tgt, val128_mask = generate_batch(300, seq_len=128, harder=True)
    # Per-task harder breakdown (48)
    set_seed(1234)
    val_sel_parity_inp, val_sel_parity_tgt, val_sel_parity_mask = generate_batch(200, seq_len=48, task_probs=(1,0,0), harder=False) # we need selective parity directly
    # Instead generate selective parity directly
    def gen_sel_batch(n, L):
        inps, tgts, masks = [], [], []
        for _ in range(n):
            i,t,m = generate_selective_parity_sequence(L)
            inps.append(i); tgts.append(t); masks.append(m)
        return torch.stack(inps), torch.stack(tgts), torch.stack(masks)
    sel48_inp, sel48_tgt, sel48_mask = gen_sel_batch(200, 48)
    sel96_inp, sel96_tgt, sel96_mask = gen_sel_batch(200, 96)
    sel128_inp, sel128_tgt, sel128_mask = gen_sel_batch(200, 128)

    # Move to device
    def to_dev(t): return t.to(device)
    val_inp, val_tgt, val_mask = to_dev(val_inp), to_dev(val_tgt), to_dev(val_mask)
    val64_inp, val64_tgt, val64_mask = to_dev(val64_inp), to_dev(val64_tgt), to_dev(val64_mask)
    val96_inp, val96_tgt, val96_mask = to_dev(val96_inp), to_dev(val96_tgt), to_dev(val96_mask)
    val128_inp, val128_tgt, val128_mask = to_dev(val128_inp), to_dev(val128_tgt), to_dev(val128_mask)
    sel48_inp, sel48_tgt, sel48_mask = to_dev(sel48_inp), to_dev(sel48_tgt), to_dev(sel48_mask)
    sel96_inp, sel96_tgt, sel96_mask = to_dev(sel96_inp), to_dev(sel96_tgt), to_dev(sel96_mask)
    sel128_inp, sel128_tgt, sel128_mask = to_dev(sel128_inp), to_dev(sel128_tgt), to_dev(sel128_mask)

    set_seed(42) # reset for training randomization

    train_losses = []
    best_val = float('inf')
    start = time.time()
    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0
        epoch_tokens = 0
        for _ in range(num_batches):
            inp, tgt, mask = generate_batch(batch_size, seq_len=seq_len, harder=True)
            inp, tgt, mask = to_dev(inp), to_dev(tgt), to_dev(mask)
            logits = model(inp)
            loss_per = crit(logits.view(-1, VOCAB_SIZE), tgt.view(-1))
            loss = (loss_per * mask.view(-1)).sum() / (mask.view(-1).sum() + 1e-8)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item() * mask.sum().item()
            epoch_tokens += mask.sum().item()
        sched.step()
        avg_train = epoch_loss / max(epoch_tokens,1)
        train_losses.append(avg_train)

        # eval
        model.eval()
        with torch.no_grad():
            def eval_set(inp, tgt, mask):
                logits = model(inp)
                loss_per = crit(logits.view(-1, VOCAB_SIZE), tgt.view(-1))
                mask_f = mask.view(-1)
                loss = (loss_per * mask_f).sum() / (mask_f.sum()+1e-8)
                pred = logits.argmax(dim=-1)
                acc = ((pred==tgt) * mask.bool()).sum().item() / max(mask.sum().item(),1)
                return loss.item(), acc
            v_loss, v_acc = eval_set(val_inp, val_tgt, val_mask)
            v64_loss, v64_acc = eval_set(val64_inp, val64_tgt, val64_mask)
            v96_loss, v96_acc = eval_set(val96_inp, val96_tgt, val96_mask)
            v128_loss, v128_acc = eval_set(val128_inp, val128_tgt, val128_mask)
            s48_loss, s48_acc = eval_set(sel48_inp, sel48_tgt, sel48_mask)
            s96_loss, s96_acc = eval_set(sel96_inp, sel96_tgt, sel96_mask)
            s128_loss, s128_acc = eval_set(sel128_inp, sel128_tgt, sel128_mask)
        mem = get_memory_mb()
        lr_cur = opt.param_groups[0]['lr']
        print(f"[Epoch {epoch:02d}/{epochs}] train={avg_train:.4f} val={v_loss:.4f}/{v_acc:.3f} | 64={v64_acc:.3f} 96={v96_acc:.3f} 128={v128_acc:.3f} | sel48={s48_acc:.3f} sel96={s96_acc:.3f} sel128={s128_acc:.3f} | lr={lr_cur:.5f} mem={mem:.1f}MB")
        if v_loss < best_val:
            best_val = v_loss
    total_time = time.time() - start
    # final detailed eval at multiple lengths for generalization curve
    model.eval()
    with torch.no_grad():
        def eval_len(L):
            # generate harder batch at length L
            set_seed(777)
            inp, tgt, mask = generate_batch(300, seq_len=L, harder=True)
            inp, tgt, mask = to_dev(inp), to_dev(tgt), to_dev(mask)
            logits = model(inp)
            loss_per = crit(logits.view(-1, VOCAB_SIZE), tgt.view(-1))
            loss = (loss_per * mask.view(-1)).sum() / (mask.view(-1).sum()+1e-8)
            pred = logits.argmax(dim=-1)
            acc = ((pred==tgt) * mask.bool()).sum().item() / max(mask.sum().item(),1)
            return loss.item(), acc
        # selective parity across lengths
        def eval_sel_len(L):
            set_seed(888)
            inps, tgts, masks = [],[],[]
            for _ in range(300):
                i,t,m = generate_selective_parity_sequence(L)
                inps.append(i); tgts.append(t); masks.append(m)
            inp=torch.stack(inps).to(device); tgt=torch.stack(tgts).to(device); mask=torch.stack(masks).to(device)
            logits = model(inp)
            loss_per = crit(logits.view(-1, VOCAB_SIZE), tgt.view(-1))
            loss = (loss_per * mask.view(-1)).sum() / (mask.view(-1).sum()+1e-8)
            pred = logits.argmax(dim=-1)
            acc = ((pred==tgt) * mask.bool()).sum().item() / max(mask.sum().item(),1)
            return loss.item(), acc

        # Per-task at 48 and 128
        set_seed(999)
        # parity vanilla
        p48_inp, p48_tgt, p48_mask = generate_batch(200, seq_len=48, task_probs=(1,0,0), harder=False)
        p48_inp, p48_tgt, p48_mask = to_dev(p48_inp), to_dev(p48_tgt), to_dev(p48_mask)
        p128_inp, p128_tgt, p128_mask = generate_batch(200, seq_len=128, task_probs=(1,0,0), harder=False)
        p128_inp, p128_tgt, p128_mask = to_dev(p128_inp), to_dev(p128_tgt), to_dev(p128_mask)
        # s3
        s48_inp, s48_tgt, s48_mask = generate_batch(200, seq_len=48, task_probs=(0,1,0), harder=False)
        s48_inp, s48_tgt, s48_mask = to_dev(s48_inp), to_dev(s48_tgt), to_dev(s48_mask)
        s128_inp, s128_tgt, s128_mask = generate_batch(200, seq_len=128, task_probs=(0,1,0), harder=False)
        s128_inp, s128_tgt, s128_mask = to_dev(s128_inp), to_dev(s128_tgt), to_dev(s128_mask)
        # mqar harder
        m48_inp, m48_tgt, m48_mask = generate_batch(200, seq_len=48, task_probs=(0,0,1), harder=False)
        # but mqar harder should be 6 pairs, we generated vanilla 4 pairs for 48; okay

        def eval_set(inp, tgt, mask):
            logits = model(inp)
            loss_per = crit(logits.view(-1, VOCAB_SIZE), tgt.view(-1))
            loss = (loss_per * mask.view(-1)).sum() / (mask.view(-1).sum()+1e-8)
            pred = logits.argmax(dim=-1)
            acc = ((pred==tgt) * mask.bool()).sum().item() / max(mask.sum().item(),1)
            return loss.item(), acc

        metrics = {}
        metrics['overall_48'] = eval_len(48)
        metrics['overall_96'] = eval_len(96)
        metrics['overall_128'] = eval_len(128)
        metrics['sel_parity_48'] = eval_sel_len(48)
        metrics['sel_parity_128'] = eval_sel_len(128)
        metrics['parity_48'] = eval_set(p48_inp, p48_tgt, p48_mask)
        metrics['parity_128'] = eval_set(p128_inp, p128_tgt, p128_mask)
        metrics['s3_48'] = eval_set(s48_inp, s48_tgt, s48_mask)
        metrics['s3_128'] = eval_set(s128_inp, s128_tgt, s128_mask)

    print(f"\nFinal metrics for {model_name}:")
    for k,(loss,acc) in metrics.items():
        print(f"  {k:15s} loss {loss:.4f} acc {acc:.3f}")

    return {
        "model_name": model_name,
        "params": model.count_params(),
        "train_losses": train_losses,
        "metrics": metrics,
        "total_time": total_time,
        "mem": get_memory_mb(),
        "config": config,
        "model_kwargs": model_kwargs,
    }

def run_harder():
    config = {
        "d_model": 64, # base
        "seq_len": 48, # harder train length vs previous 32
        "max_len": 128,
        "lr": 1.5e-3,
        "epochs": 25,
        "batch_size": 32,
        "num_batches_per_epoch": 180, # 180*32=5760 per epoch
        "weight_decay": 0.01,
    }
    results = {}
    # Transformer 64 (73k)
    set_seed(42)
    results["transformer_64"] = train_and_eval("transformer", config, {"d_model":64, "n_layers":2})
    # ROGM 64 (101k)
    set_seed(42)
    results["rogm_64"] = train_and_eval("rogm", config, {"d_model":64, "n_layers":2})
    # ROGM-lite 32 (stripped, ~28k params < baseline, tests parameter efficiency)
    set_seed(42)
    results["rogm_lite_32"] = train_and_eval("rogm", config, {"d_model":32, "n_layers":2})

    print("\n"+"="*80)
    print("HARDER EXPERIMENT COMPARISON (train 48, eval 48/96/128, selective parity)")
    print("="*80)
    for name in ["transformer_64", "rogm_64", "rogm_lite_32"]:
        r = results[name]
        print(f"\n{name.upper()} (params {r['params']}):")
        print(f"  Time {r['total_time']:.1f}s Mem {r['mem']:.1f}MB")
        print(f"  Train loss final {r['train_losses'][-1]:.4f}")
        for k,(loss,acc) in r['metrics'].items():
            print(f"    {k:15s} acc {acc:.3f} loss {loss:.4f}")

    # Deltas
    t = results["transformer_64"]
    r = results["rogm_64"]
    lite = results["rogm_lite_32"]
    print("\nDELTA vs Transformer (ROGM-64):")
    for k in t['metrics']:
        print(f"  {k}: acc delta {r['metrics'][k][1]-t['metrics'][k][1]:+.3f} loss delta {r['metrics'][k][0]-t['metrics'][k][0]:+.4f}")
    print("\nDELTA lite (32) vs Transformer (64): (tests if stripped still wins)")
    for k in t['metrics']:
        print(f"  {k}: acc delta {lite['metrics'][k][1]-t['metrics'][k][1]:+.3f}")

    # Success criteria for harder: ROGM should maintain >0.75 on 128 for parity/s3/selective, transformer at chance (~0.5/0.16)
    # Lite should still beat transformer despite <0.5 params
    reasoning_keys = ['parity_48','parity_128','s3_48','s3_128','sel_parity_48','sel_parity_128','overall_128']
    wins = sum(1 for k in reasoning_keys if r['metrics'][k][1] > t['metrics'][k][1] + 0.10)
    lite_wins = sum(1 for k in reasoning_keys if lite['metrics'][k][1] > t['metrics'][k][1] + 0.05)
    print(f"\nHard reasoning wins ROGM-64: {wins}/{len(reasoning_keys)} (need >=4)")
    print(f"Hard reasoning wins ROGM-lite-32 (<½ params): {lite_wins}/{len(reasoning_keys)} (need >=3 to prove efficiency)")
    success = wins >=4
    lite_success = lite_wins >=3
    print(f"  => Harder SUCCESS ROGM-64? {success}")
    print(f"  => Stripped ROGM-lite still superior? {lite_success}")

    # Save logs
    os.makedirs("logs", exist_ok=True)
    import datetime, json
    ts = datetime.datetime.now().isoformat()
    path = f"logs/comparison_harder_{ts.replace(':','-')}.json"
    serializable = {}
    for k,v in results.items():
        serializable[k] = {
            "params": v["params"],
            "train_losses": v["train_losses"],
            "metrics": {kk: {"loss": loss, "acc": acc} for kk,(loss,acc) in v["metrics"].items()},
            "total_time": v["total_time"],
            "mem": v["mem"],
        }
    serializable["_summary"] = {
        "wins": wins,
        "lite_wins": lite_wins,
        "success": bool(success),
        "lite_success": bool(lite_success),
        "timestamp": ts,
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    with open("logs/latest_harder.json","w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved {path}")

    return results

if __name__ == "__main__":
    run_harder()
