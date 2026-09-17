import torch, torch.nn as nn, time, os, random, json, psutil
from src.dataset import generate_batch, VOCAB_SIZE, generate_selective_parity_sequence
from src.models_ablation import get_ablation_model
torch.set_num_threads(2); torch.set_num_interop_threads(2)
device=torch.device("cpu")
def set_seed(s): random.seed(s); torch.manual_seed(s)
def mem(): return psutil.Process(os.getpid()).memory_info().rss/1024/1024
def eval_model(model, crit):
    model.eval()
    with torch.no_grad():
        def gen_sel(n,L):
            a,b,c=[],[],[]
            for _ in range(n):
                i,t,m=generate_selective_parity_sequence(L)
                a.append(i); b.append(t); c.append(m)
            return torch.stack(a).to(device), torch.stack(b).to(device), torch.stack(c).to(device)
        def ev(inp,tgt,mask):
            logits=model(inp); loss=(crit(logits.view(-1,VOCAB_SIZE),tgt.view(-1))*mask.view(-1)).sum()/(mask.view(-1).sum()+1e-8)
            pred=logits.argmax(-1); acc=((pred==tgt)*mask.bool()).sum().item()/max(mask.sum().item(),1)
            return loss.item(),acc
        metrics={}
        for L in [48,128]:
            set_seed(800+L); inp,tgt,mask=generate_batch(250, seq_len=L, harder=True)
            inp,tgt,mask=inp.to(device),tgt.to(device),mask.to(device)
            metrics[f'overall_{L}']=ev(inp,tgt,mask)
            set_seed(900+L); inp,tgt,mask=gen_sel(200,L)
            metrics[f'sel_parity_{L}']=ev(inp,tgt,mask)
            set_seed(1000+L); inp2,tgt2,mask2=generate_batch(200, seq_len=L, task_probs=(1,0,0), harder=False)
            inp2,tgt2,mask2=inp2.to(device),tgt2.to(device),mask2.to(device)
            metrics[f'parity_{L}']=ev(inp2,tgt2,mask2)
            set_seed(1100+L); inp3,tgt3,mask3=generate_batch(200, seq_len=L, task_probs=(0,1,0), harder=False)
            inp3,tgt3,mask3=inp3.to(device),tgt3.to(device),mask3.to(device)
            metrics[f's3_{L}']=ev(inp3,tgt3,mask3)
        return metrics

def train_variant(name, model, cfg):
    print(f"\nTRAINING {name} params {model.count_params()}")
    model.to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'])
    crit=nn.CrossEntropyLoss(reduction='none')
    start=time.time()
    for epoch in range(1,cfg['epochs']+1):
        model.train()
        el=0; toks=0
        for _ in range(cfg['num_batches_per_epoch']):
            inp,tgt,mask=generate_batch(cfg['batch_size'], seq_len=cfg['seq_len'], harder=True)
            inp,tgt,mask=inp.to(device),tgt.to(device),mask.to(device)
            logits=model(inp)
            loss=(crit(logits.view(-1,VOCAB_SIZE),tgt.view(-1))*mask.view(-1)).sum()/(mask.view(-1).sum()+1e-8)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            el+=loss.item()*mask.sum().item(); toks+=mask.sum().item()
        sched.step()
        # quick val
        model.eval()
        with torch.no_grad():
            set_seed(777); vinp,vtgt,vmask=generate_batch(200, seq_len=cfg['seq_len'], harder=True)
            vinp,vtgt,vmask=vinp.to(device),vtgt.to(device),vmask.to(device)
            logits=model(vinp)
            vloss=(crit(logits.view(-1,VOCAB_SIZE),vtgt.view(-1))*vmask.view(-1)).sum()/(vmask.view(-1).sum()+1e-8)
            pred=logits.argmax(-1); vacc=((pred==vtgt)*vmask.bool()).sum().item()/max(vmask.sum().item(),1)
        print(f"Epoch {epoch}/{cfg['epochs']} train {el/max(toks,1):.4f} val {vloss.item():.4f}/{vacc:.3f} mem {mem():.1f}")
    metrics=eval_model(model, crit)
    print(f"Final {name}:")
    for k,(l,a) in metrics.items(): print(f"  {k} acc {a:.3f}")
    return {"name":name,"params":model.count_params(),"metrics":metrics,"time":time.time()-start}

cfg={"seq_len":48,"max_len":128,"lr":1.5e-3,"epochs":15,"batch_size":32,"num_batches_per_epoch":120}
results={}
import random
for var in ["noMem","minimal","lite"]:
    set_seed(42)
    m=get_ablation_model(var, d_model=64 if var!="lite" else 32, max_len=128)
    results[var]=train_variant(f"rogm_{var}", m, cfg)

# also include previous full/singleH/noOsc from earlier run for comparison
# load previous partial
try:
    import json
    with open("logs/ablation_2026-09-17T05-36-unknown.json") as f:
        pass
except: pass
print("\nQUICK ABLATION DONE")
for k,v in results.items():
    print(k, v["metrics"]["overall_48"], v["metrics"]["s3_128"])
os.makedirs("logs",exist_ok=True)
with open("logs/quick_ablation.json","w") as f:
    json.dump({k:{"params":v["params"],"metrics":{kk:{"acc":a,"loss":l} for kk,(l,a) in v["metrics"].items()}} for k,v in results.items()}, f, indent=2)
