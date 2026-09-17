"""
S5 Permutation Composition Dataset - Non-solvable group, ultimate state-tracking test
Merrill et al. proves SSMs/Transformers cannot solve S5 (TC0 limit). ROGM with orthogonal group should.
S5 has 120 elements (permutations of 0..4). Vocab size 128 to accommodate.
"""
import torch, random, itertools

VOCAB_S5_SIZE = 128
S5_OFFSET = 0  # tokens 0..119 are S5 elements, 120..127 are special

def build_s5():
    perms = list(itertools.permutations(range(5))) # 120
    perm_to_idx = {p:i for i,p in enumerate(perms)}
    comp = [[0]*120 for _ in range(120)]
    for i,a in enumerate(perms):
        for j,b in enumerate(perms):
            c = tuple(b[a[k]] for k in range(5))
            comp[i][j] = perm_to_idx[c]
    return perms, comp

S5_PERMS, S5_COMP = build_s5()
print(f"S5 built: {len(S5_PERMS)} elements")

def generate_s5_sequence(seq_len):
    # random S5 tokens 0..119
    inp = torch.randint(0,120,(seq_len,), dtype=torch.long)
    target = torch.empty_like(inp)
    cur = inp[0].item()
    target[0]=cur
    for i in range(1,seq_len):
        cur = S5_COMP[cur][inp[i].item()]
        target[i]=cur
    mask = torch.ones(seq_len, dtype=torch.float32)
    return inp, target, mask

def generate_batch_s5(batch_size, seq_len=32):
    inps,tgts,masks=[],[],[]
    for _ in range(batch_size):
        inp,tgt,m = generate_s5_sequence(seq_len)
        inps.append(inp); tgts.append(tgt); masks.append(m)
    return torch.stack(inps), torch.stack(tgts), torch.stack(masks)

if __name__=="__main__":
    inp,tgt,m = generate_s5_sequence(8)
    print(inp.tolist())
    print(tgt.tolist())
    # test compose
    # check random
    import random
    for _ in range(3):
        a,b=random.randint(0,119), random.randint(0,119)
        c=S5_COMP[a][b]
        # verify via permutation
        pa=S5_PERMS[a]; pb=S5_PERMS[b]
        pc=tuple(pb[pa[k]] for k in range(5))
        assert S5_PERMS[c]==pc
    print("S5 compose ok")
    batch_inp,batch_tgt,batch_mask=generate_batch_s5(2,16)
    print(batch_inp.shape)
