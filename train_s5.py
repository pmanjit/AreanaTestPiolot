"""
Iteration 4: S5 Non-Solvable Group — Ultimate State-Tracking Challenge
Merrill proves SSM/Transformer in TC0 cannot solve S5 (non-solvable). ROGM with O(d) Householder should.
Vocab 128, S5 120 elements, sequence composition.
Micro-scale: d=64, 2 layers, train 32 → eval 64/96/128.
"""
import torch, torch.nn as nn, time, os, random, json, psutil
from src.dataset_s5 import generate_batch_s5, VOCAB_S5_SIZE
from src.models import MicroTransformer
from src.models_ablation import ROGMAblationModel

torch.set_num_threads(2); torch.set_num_interop_threads(2)
device=torch.device("cpu")
def set_seed(s): random.seed(s); torch.manual_seed(s)
def mem(): return psutil.Process(os.getpid()).memory_info().rss/1024/1024

def get_transformer_s5(d_model=64, max_len=128):
    m=MicroTransformer(vocab_size=VOCAB_S5_SIZE, d_model=d_model, n_layers=2, max_len=max_len, dropout=0.1)
    return m

def get_rogm_s5(d_model=64, max_len=128):
    m=ROGMAblationModel(vocab_size=VOCAB_S5_SIZE, d_model=d_model, n_layers=2, max_len=max_len, use_double=True, use_osc=True, use_memory=True)
    return m

def train_s5(model_name, model, cfg):
    print(f"\n{'='*70}\nTRAINING {model_name} params {sum(p.numel() for p in model.parameters())} {cfg}\n{'='*70}")
    model.to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'])
    crit=nn.CrossEntropyLoss(reduction='none')
    best=float('inf')
    losses=[]
    for epoch in range(1,cfg['epochs']+1):
        model.train()
        el=0; toks=0
        for _ in range(cfg['num_batches_per_epoch']):
            inp,tgt,mask=generate_batch_s5(cfg['batch_size'], seq_len=cfg['seq_len'])
            inp,tgt,mask=inp.to(device),tgt.to(device),mask.to(device)
            logits=model(inp)
            loss=(crit(logits.view(-1,VOCAB_S5_SIZE), tgt.view(-1))*mask.view(-1)).sum()/(mask.view(-1).sum()+1e-8)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            el+=loss.item()*mask.sum().item(); toks+=mask.sum().item()
        sched.step()
        avg=el/max(toks,1); losses.append(avg)
        # val
        model.eval()
        with torch.no_grad():
            set_seed(777); vinp,vtgt,vmask=generate_batch_s5(300, seq_len=cfg['seq_len'])
            vinp,vtgt,vmask=vinp.to(device),vtgt.to(device),vmask.to(device)
            logits=model(vinp)
            vloss=(crit(logits.view(-1,VOCAB_S5_SIZE), vtgt.view(-1))*vmask.view(-1)).sum()/(vmask.view(-1).sum()+1e-8)
            pred=logits.argmax(-1); acc=((pred==vtgt)*vmask.bool()).sum().item()/max(vmask.sum().item(),1)
        print(f"Epoch {epoch}/{cfg['epochs']} train {avg:.4f} val {vloss.item():.4f}/{acc:.3f} lr {opt.param_groups[0]['lr']:.5f} mem {mem():.1f}")
        if vloss<best: best=vloss.item()
    # eval at lengths
    model.eval()
    metrics={}
    with torch.no_grad():
        for L in [32,64,96,128]:
            set_seed(900+L); inp,tgt,mask=generate_batch_s5(250, seq_len=L)
            inp,tgt,mask=inp.to(device),tgt.to(device),mask.to(device)
            logits=model(inp)
            loss=(crit(logits.view(-1,VOCAB_S5_SIZE), tgt.view(-1))*mask.view(-1)).sum()/(mask.view(-1).sum()+1e-8)
            pred=logits.argmax(-1); acc=((pred==tgt)*mask.bool()).sum().item()/max(mask.sum().item(),1)
            metrics[f's5_{L}']=(loss.item(), acc)
            # chance is 1/120 =0.0083
        # also test parity for control? not needed
    print(f"Final {model_name}:")
    for k,(l,a) in metrics.items():
        print(f"  {k} loss {l:.4f} acc {a:.3f} (chance 0.008)")
    return {"name":model_name,"params":sum(p.numel() for p in model.parameters()),"losses":losses,"metrics":metrics,"time":time.time()-time.time()}

def run_s5():
    cfg={"seq_len":32,"max_len":128,"lr":1.2e-3,"epochs":20,"batch_size":32,"num_batches_per_epoch":160}
    results={}
    # Transformer
    set_seed(42)
    trans=get_transformer_s5(d_model=64, max_len=128)
    results["transformer"]=train_model("transformer_s5", trans, cfg)
    # ROGM full
    set_seed(42)
    rogm=get_rogm_s5(d_model=64, max_len=128)
    results["rogm"]=train_model("rogm_s5", rogm, cfg)
    # lite 32
    set_seed(42)
    lite=get_rogm_s5(d_model=32, max_len=128)
    results["rogm_lite32"]=train_model("rogm_lite32", lite, cfg)
    return results

def train_model(name, model, cfg):
    # wrapper to keep same as above but with timing
    start=time.time()
    print(f"\nTRAINING {name} params {sum(p.numel() for p in model.parameters())}")
    model.to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'])
    crit=nn.CrossEntropyLoss(reduction='none')
    losses=[]
    for epoch in range(1,cfg['epochs']+1):
        model.train()
        el=0; toks=0
        for _ in range(cfg['num_batches_per_epoch']):
            inp,tgt,mask=generate_batch_s5(cfg['batch_size'], seq_len=cfg['seq_len'])
            inp,tgt,mask=inp.to(device),tgt.to(device),mask.to(device)
            logits=model(inp)
            loss=(crit(logits.view(-1,VOCAB_S5_SIZE), tgt.view(-1))*mask.view(-1)).sum()/(mask.view(-1).sum()+1e-8)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            el+=loss.item()*mask.sum().item(); toks+=mask.sum().item()
        sched.step()
        avg=el/max(toks,1); losses.append(avg)
        model.eval()
        with torch.no_grad():
            set_seed(777); vinp,vtgt,vmask=generate_batch_s5(300, seq_len=cfg['seq_len'])
            vinp,vtgt,vmask=vinp.to(device),vtgt.to(device),vmask.to(device)
            logits=model(vinp)
            vloss=(crit(logits.view(-1,VOCAB_S5_SIZE), vtgt.view(-1))*vmask.view(-1)).sum()/(vmask.view(-1).sum()+1e-8)
            pred=logits.argmax(-1); acc=((pred==vtgt)*vmask.bool()).sum().item()/max(vmask.sum().item(),1)
        print(f"Epoch {epoch}/{cfg['epochs']} train {avg:.4f} val {vloss.item():.4f}/{acc:.3f} mem {mem():.1f}")
    # eval
    model.eval()
    metrics={}
    with torch.no_grad():
        for L in [32,64,96,128]:
            set_seed(900+L); inp,tgt,mask=generate_batch_s5(250, seq_len=L)
            inp,tgt,mask=inp.to(device),tgt.to(device),mask.to(device)
            logits=model(inp)
            loss=(crit(logits.view(-1,VOCAB_S5_SIZE), tgt.view(-1))*mask.view(-1)).sum()/(mask.view(-1).sum()+1e-8)
            pred=logits.argmax(-1); acc=((pred==tgt)*mask.bool()).sum().item()/max(mask.sum().item(),1)
            metrics[f's5_{L}']=(loss.item(), acc)
    print(f"Final {name}:")
    for k,(l,a) in metrics.items():
        print(f"  {k} acc {a:.3f} chance 0.008")
    return {"name":name,"params":sum(p.numel() for p in model.parameters()),"losses":losses,"metrics":metrics,"time":time.time()-start, "mem":mem()}

if __name__=="__main__":
    cfg={"seq_len":32,"max_len":128,"lr":1.2e-3,"epochs":20,"batch_size":32,"num_batches_per_epoch":160}
    import time, random, os, json
    results={}
    set_seed= lambda s: (random.seed(s), torch.manual_seed(s))
    # Actually define properly
    def set_seed2(s):
        random.seed(s); torch.manual_seed(s)
    # run
    set_seed2(42)
    trans=get_transformer_s5(d_model=64, max_len=128)
    results["transformer"]=train_model("transformer_s5", trans, cfg)
    set_seed2(42)
    rogm=get_rogm_s5(d_model=64, max_len=128)
    results["rogm"]=train_model("rogm_s5", rogm, cfg)
    set_seed2(42)
    lite=get_rogm_s5(d_model=32, max_len=128)
    results["lite"]=train_model("rogm_lite_s5", lite, cfg)

    print("\n"+"="*80)
    print("S5 COMPARISON (non-solvable, chance 0.008)")
    print("="*80)
    for k in ["transformer","rogm","lite"]:
        r=results[k]
        print(f"\n{k} {r['params']} params:")
        for L in [32,64,96,128]:
            print(f"  s5_{L} acc {r['metrics'][f's5_{L}'][1]:.3f}")

    os.makedirs("logs",exist_ok=True)
    import datetime
    ts=datetime.datetime.now().isoformat()
    path=f"logs/s5_{ts.replace(':','-')}.json"
    serializable={k:{"params":v["params"],"metrics":{kk:{"loss":l,"acc":a} for kk,(l,a) in v["metrics"].items()}} for k,v in results.items()}
    with open(path,"w") as f: json.dump(serializable,f,indent=2)
    with open("logs/latest_s5.json","w") as f: json.dump(serializable,f,indent=2)
    print(f"Saved {path}")
