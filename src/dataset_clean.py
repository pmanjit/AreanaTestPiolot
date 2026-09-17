"""
Synthetic reasoning dataset generator for micro-scale proof-of-concept.
Tasks: Parity (C2), S3 permutation composition, MQAR ( associative recall )
Vocab: 16 (0-1 parity, 2-7 S3, 8-11 keys, 12-15 values)
Fits in <2GB RAM, generates on the fly.
"""
import torch
import random
import itertools
import math

VOCAB_SIZE = 16

def build_s3():
    perms = list(itertools.permutations([0,1,2]))
    perm_to_idx = {p:i for i,p in enumerate(perms)}
    comp = [[0]*6 for _ in range(6)]
    for i,a in enumerate(perms):
        for j,b in enumerate(perms):
            c = tuple(b[a[k]] for k in range(3))
            comp[i][j] = perm_to_idx[c]
    return perms, comp, perm_to_idx

S3_PERMS, S3_COMP, _ = build_s3()

def s3_compose(a_id, b_id):
    a = a_id - 2
    b = b_id - 2
    return S3_COMP[a][b] + 2

S3_COMP_TOKEN = [[0]*16 for _ in range(16)]
for a in range(2,8):
    for b in range(2,8):
        S3_COMP_TOKEN[a][b] = s3_compose(a,b)

def generate_parity_sequence(seq_len):
    inp = torch.randint(0,2,(seq_len,), dtype=torch.long)
    target = torch.empty_like(inp)
    cur = 0
    for i, b in enumerate(inp.tolist()):
        cur ^= b
        target[i] = cur
    mask = torch.ones(seq_len, dtype=torch.float32)
    return inp, target, mask

def generate_s3_sequence(seq_len):
    inp = torch.randint(2,8,(seq_len,), dtype=torch.long)
    target = torch.empty_like(inp)
    cur = inp[0].item()
    target[0] = cur
    for i in range(1, seq_len):
        cur = S3_COMP_TOKEN[cur][inp[i].item()]
        target[i] = cur
    mask = torch.ones(seq_len, dtype=torch.float32)
    return inp, target, mask

def generate_mqar_sequence(seq_len=32, num_pairs=4, num_queries=4):
    assert seq_len >= num_pairs*2 + num_queries
    filler_len = seq_len - num_pairs*2 - num_queries
    keys_avail = [8,9,10,11]
    values_avail = [12,13,14,15]
    selected_keys = random.sample(keys_avail, num_pairs)
    selected_vals = [random.choice(values_avail) for _ in range(num_pairs)]
    kv_map = dict(zip(selected_keys, selected_vals))
    kvs = list(zip(selected_keys, selected_vals))
    random.shuffle(kvs)
    storage = []
    for k,v in kvs:
        storage.append(k)
        storage.append(v)
    filler = torch.randint(0, VOCAB_SIZE, (filler_len,), dtype=torch.long).tolist() if filler_len>0 else []
    query_keys = [random.choice(selected_keys) for _ in range(num_queries)]
    inp_list = storage + filler + query_keys
    assert len(inp_list) == seq_len
    inp = torch.tensor(inp_list, dtype=torch.long)
    target = torch.empty_like(inp)
    mask = torch.zeros(seq_len, dtype=torch.float32)
    target[:] = inp[:]
    query_start = num_pairs*2 + filler_len
    for i, qk in enumerate(query_keys):
        pos = query_start + i
        target[pos] = kv_map[qk]
        mask[pos] = 1.0
    return inp, target, mask

def generate_selective_parity_sequence(seq_len):
    inp = torch.empty(seq_len, dtype=torch.long)
    target = torch.empty(seq_len, dtype=torch.long)
    mask = torch.ones(seq_len, dtype=torch.float32)
    cur = 0
    for i in range(seq_len):
        if random.random() < 0.3:
            inp[i] = 15
            target[i] = cur
        else:
            bit = random.randint(0,1)
            inp[i] = bit
            cur ^= bit
            target[i] = cur
    return inp, target, mask

def generate_harder_mqar_sequence(seq_len=48, num_pairs=4, num_queries=4):
    return generate_mqar_sequence(seq_len=seq_len, num_pairs=num_pairs, num_queries=num_queries)

def generate_batch(batch_size, seq_len=32, task_probs=(0.4,0.3,0.3), seq_len_variable=False, harder=False):
    if not harder:
        inps, tgts, masks = [], [], []
        for _ in range(batch_size):
            r = random.random()
            if r < task_probs[0]:
                inp, tgt, m = generate_parity_sequence(seq_len)
            elif r < task_probs[0] + task_probs[1]:
                inp, tgt, m = generate_s3_sequence(seq_len)
            else:
                inp, tgt, m = generate_mqar_sequence(seq_len=seq_len)
            inps.append(inp); tgts.append(tgt); masks.append(m)
        return torch.stack(inps), torch.stack(tgts), torch.stack(masks)
    else:
        inps, tgts, masks = [], [], []
        for _ in range(batch_size):
            r = random.random()
            if r < 0.30:
                inp, tgt, m = generate_selective_parity_sequence(seq_len)
            elif r < 0.55:
                inp, tgt, m = generate_s3_sequence(seq_len)
            elif r < 0.80:
                inp, tgt, m = generate_harder_mqar_sequence(seq_len=seq_len, num_pairs=4, num_queries=4)
            else:
                inp, tgt, m = generate_parity_sequence(seq_len)
            inps.append(inp); tgts.append(tgt); masks.append(m)
        return torch.stack(inps), torch.stack(tgts), torch.stack(masks)

def generate_validation_set(num_samples=2000, seq_len=32, task_probs=(0.4,0.3,0.3), harder=False):
    if harder:
        seq_len = 64
    return generate_batch(num_samples, seq_len=seq_len, task_probs=task_probs)

class SyntheticDataLoader:
    def __init__(self, batch_size=32, seq_len=32, num_batches_per_epoch=200, task_probs=(0.4,0.3,0.3)):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_batches_per_epoch = num_batches_per_epoch
        self.task_probs = task_probs
    def __iter__(self):
        for _ in range(self.num_batches_per_epoch):
            yield generate_batch(self.batch_size, self.seq_len, self.task_probs)
    def __len__(self):
        return self.num_batches_per_epoch

def test_dataset():
    print("Testing dataset generation...")
    for name, gen in [("parity", generate_parity_sequence), ("s3", generate_s3_sequence)]:
        inp, tgt, mask = gen(8)
        print(f"{name}: inp={inp.tolist()} tgt={tgt.tolist()} mask={mask.tolist()}")
    inp, tgt, mask = generate_mqar_sequence(16)
    print(f"mqar: inp={inp.tolist()} tgt={tgt.tolist()} mask={mask.tolist()}")
    inp, tgt, mask = generate_selective_parity_sequence(16)
    print(f"sel_parity: inp={inp.tolist()} tgt={tgt.tolist()} mask={mask.tolist()}")
    batch_inp, batch_tgt, batch_mask = generate_batch(4, 16)
    print(f"batch: inp shape {batch_inp.shape} tgt {batch_tgt.shape} mask sum {batch_mask.sum(dim=1).tolist()}")
    batch_inp, batch_tgt, batch_mask = generate_batch(4, 16, harder=True)
    print(f"batch harder: inp shape {batch_inp.shape} tgt {batch_tgt.shape} mask sum {batch_mask.sum(dim=1).tolist()}")
    print(f"S3 perms {S3_PERMS}")

if __name__ == "__main__":
    test_dataset()
