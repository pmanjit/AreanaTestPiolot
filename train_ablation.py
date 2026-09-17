"""
Iteration 3: Ablation Study — Stripping Redundant Parameters
Tests which components of ROGM are essential for superior reasoning.
Variants: full (doubleH+osc+mem), singleH (1 Householder), noOsc, noMem, minimal (singleH+noOsc)
Baseline: Transformer-64
Dataset: Harder (selective parity, S3, MQAR) train 48, eval 48/96/128
"""
import torch, torch.nn as nn, time, os, random, json, datetime, psutil
from src.dataset import generate_batch, VOCAB_SIZE, generate_selective_parity_sequence
from src.models import get_model as get_transformer
from src.models_ablation import get_ablation_model

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
device = torch.device("cpu")

def set_seed(s): 
    random.seed(s); torch.manual_seed(s)

def mem_mb(): return psutil.Process(os.getpid()).memory_info().rss/1024/1024

def evaluate_model(model, crit):
    model.eval()
    with torch.no_grad():
        def gen_sel(n,L):
            inps,tgts,masks=[],[],[]
            for _ in range(n):
                i,t,m=generate_selective_parity_sequence(L)
                inps.append(i); tgts.append(t); masks.append(m)
            return torch.stack(inps).to(device), torch.stack(tgts).to(device), torch.stack(masks).to(device)
        def eval_set(inp,tgt,mask):
            logits=model(inp)
            loss=(crit(logits.view(-1,VOCAB_SIZE),tgt.view(-1))*mask.view(-1)).sum()/(mask.view(-1).sum()+1e-8)
            pred=logits.argmax(-1)
            acc=((pred==tgt)*mask.bool()).sum().item()/max(mask.sum().item(),1)
            return loss.item(),acc
        # generate eval sets
        set_seed(777)
        # overall harder at 48/96/128
        metrics={}
        for L in [48,96,128]:
            set_seed(800+L)
            inp,tgt,mask=generate_batch(250, seq_len=L, harder=True)
            inp,tgt,mask=inp.to(device),tgt.to(device),mask.to(device)
            metrics[f'overall_{L}']=eval_set(inp,tgt,mask)
        for L in [48,128]:
            set_seed(900+L)
            inp,tgt,mask=gen_sel(200,L)
            metrics[f'sel_parity_{L}']=eval_set(inp,tgt,mask)
            # vanilla parity
            set_seed(1000+L)
            inp2,tgt2,mask2=generate_batch(200, seq_len=L, task_probs=(1,0,0), harder=False)
            inp2,tgt2,mask2=inp2.to(device),tgt2.to(device),mask2.to(device)
            metrics[f'parity_{L}']=eval_set(inp2,tgt2,mask2)
            # s3
            set_seed(1100+L)
            inp3,tgt3,mask3=generate_batch(200, seq_len=L, task_probs=(0,1,0), harder=False)
            inp3,tgt3,mask3=inp3.to(device),tgt3.to(device),mask3.to(device)
            metrics[f's3_{L}']=eval_set(inp3,tgt3,mask3)
            # mqar
            set_seed(1200+L)
            inp4,tgt4,mask4=generate_batch(200, seq_len=L, task_probs=(0,0,1), harder=False)
            inp4,tgt4,mask4=inp4.to(device),tgt4.to(device),mask4.to(device)
            metrics[f'mqar_{L}']=eval_set(inp4,tgt4,mask4)
        return metrics

def train_one_variant(name, model, config):
    print(f"\n{'='*70}\nTRAINING {name} | params {model.count_params()} | {config}\n{'='*70}")
    model.to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['epochs'])
    crit=nn.CrossEntropyLoss(reduction='none')
    best= float('inf')
    train_losses=[]
    start=time.time()
    for epoch in range(1, config['epochs']+1):
        model.train()
        epoch_loss=0; tokens=0
        for _ in range(config['num_batches_per_epoch']):
            inp,tgt,mask=generate_batch(config['batch_size'], seq_len=config['seq_len'], harder=True)
            inp,tgt,mask=inp.to(device),tgt.to(device),mask.to(device)
            logits=model(inp)
            loss=(crit(logits.view(-1,VOCAB_SIZE),tgt.view(-1))*mask.view(-1)).sum()/(mask.view(-1).sum()+1e-8)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            epoch_loss+=loss.item()*mask.sum().item(); tokens+=mask.sum().item()
        sched.step()
        avg=epoch_loss/max(tokens,1); train_losses.append(avg)
        # quick val
        model.eval()
        with torch.no_grad():
            set_seed(999)
            vinp,vtgt,vmask=generate_batch(300, seq_len=config['seq_len'], harder=True)
            vinp,vtgt,vmask=vinp.to(device),vtgt.to(device),vmask.to(device)
            logits=model(vinp)
            vloss=(crit(logits.view(-1,VOCAB_SIZE),vtgt.view(-1))*vmask.view(-1)).sum()/(vmask.view(-1).sum()+1e-8)
            pred=logits.argmax(-1)
            vacc=((pred==vtgt)*vmask.bool()).sum().item()/max(vmask.sum().item(),1)
        print(f"[Epoch {epoch:02d}/{config['epochs']}] train {avg:.4f} val {vloss.item():.4f}/{vacc:.3f} lr {opt.param_groups[0]['lr']:.5f} mem {mem_mb():.1f}MB")
        if vloss<best: best=vloss.item()
    metrics=evaluate_model(model, crit)
    print(f"\nFinal {name} metrics:")
    for k,(l,a) in metrics.items():
        print(f"  {k:15s} loss {l:.4f} acc {a:.3f}")
    return {"name":name,"params":model.count_params(),"train_losses":train_losses,"metrics":metrics,"time":time.time()-start,"mem":mem_mb()}

def run_ablation():
    config={"d_model":64,"seq_len":48,"max_len":128,"lr":1.5e-3,"epochs":22,"batch_size":32,"num_batches_per_epoch":160}
    results={}
    # Transformer baseline
    set_seed(42)
    trans=get_transformer("transformer", vocab_size=VOCAB_SIZE, d_model=64, n_layers=2, max_len=128)
    results["transformer"]=train_one_variant("transformer", trans, config)
    # Full ROGM
    set_seed(42)
    full=get_ablation_model("full", d_model=64, max_len=128)
    results["rogm_full"]=train_one_variant("rogm_full", full, config)
    # Single Householder (remove second)
    set_seed(42)
    singleH=get_ablation_model("singleH", d_model=64, max_len=128)
    results["rogm_singleH"]=train_one_variant("rogm_singleH", singleH, config)
    # No Oscillator
    set_seed(42)
    noOsc=get_ablation_model("noOsc", d_model=64, max_len=128)
    results["rogm_noOsc"]=train_one_variant("rogm_noOsc", noOsc, config)
    # No Memory
    set_seed(42)
    noMem=get_ablation_model("noMem", d_model=64, max_len=128)
    results["rogm_noMem"]=train_one_variant("rogm_noMem", noMem, config)
    # Minimal (singleH+noOsc, keep memory) — tests if minimal orthogonal+memory suffices
    set_seed(42)
    minimal=get_ablation_model("minimal", d_model=64, max_len=128)
    results["rogm_minimal"]=train_one_variant("rogm_minimal", minimal, config)
    # Lite (d32 full) for efficiency reference
    set_seed(42)
    lite=get_ablation_model("lite", d_model=64, max_len=128) # actually d32 inside
    results["rogm_lite32"]=train_one_variant("rogm_lite32", lite, config)

    print("\n"+"="*90)
    print("ABLATION COMPARISON (train 48, eval 48/128)")
    print("="*90)
    order=["transformer","rogm_full","rogm_singleH","rogm_noOsc","rogm_noMem","rogm_minimal","rogm_lite32"]
    for n in order:
        r=results[n]
        print(f"\n{n.upper()} ({r['params']} params, {r['time']:.1f}s):")
        for k in ["overall_48","overall_128","sel_parity_48","sel_parity_128","parity_48","parity_128","s3_48","s3_128","mqar_48","mqar_128"]:
            if k in r["metrics"]:
                print(f"  {k:15s} acc {r['metrics'][k][1]:.3f}")

    # Analyze
    base=results["transformer"]
    full=results["rogm_full"]
    print("\n--- DELTA vs Transformer (full) ---")
    for k in base["metrics"]:
        print(f"  {k}: {full['metrics'][k][1]-base['metrics'][k][1]:+.3f}")
    print("\n--- ABLATION DROPS vs Full (negative means component needed) ---")
    for variant in ["rogm_singleH","rogm_noOsc","rogm_noMem","rogm_minimal"]:
        print(f"\n{variant} vs full:")
        for k in ["parity_128","s3_128","sel_parity_128","overall_128","mqar_128"]:
            delta=results[variant]["metrics"][k][1]-full["metrics"][k][1]
            print(f"  {k} {delta:+.3f} {'(CRITICAL)' if delta<-0.10 else ''}")

    # Success criteria: full should win >0.10 on reasoning; ablations that drop >0.15 indicate essential component
    reasoning=["parity_128","s3_128","sel_parity_128"]
    wins=sum(1 for k in reasoning if full["metrics"][k][1] > base["metrics"][k][1]+0.10)
    print(f"\nFull wins vs Transformer on reasoning (parity/s3/sel): {wins}/3")
    # Identify essential components
    for variant, desc in [("rogm_singleH","Second Householder"),("rogm_noOsc","Oscillator"),("rogm_noMem","Holographic Memory")]:
        drops=[results[variant]["metrics"][k][1]-full["metrics"][k][1] for k in reasoning]
        print(f"{desc} ablation avg drop {sum(drops)/len(drops):+.3f} -> {'ESSENTIAL' if min(drops)<-0.10 else 'redundant'}")

    # Save
    os.makedirs("logs", exist_ok=True)
    import datetime as dt
    ts=dt.datetime.now().isoformat()
    path=f"logs/ablation_{ts.replace(':','-')}.json"
    serializable={}
    for k,v in results.items():
        serializable[k]={"params":v["params"],"train_losses":v["train_losses"],"metrics":{kk:{"loss":l,"acc":a} for kk,(l,a) in v["metrics"].items()},"time":v["time"],"mem":v["mem"]}
    serializable["_summary"]={"timestamp":ts}
    with open(path,"w") as f: json.dump(serializable,f,indent=2)
    with open("logs/latest_ablation.json","w") as f: json.dump(serializable,f,indent=2)
    print(f"Saved {path}")
    return results

if __name__=="__main__":
    run_ablation()
