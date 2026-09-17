"""
Models: Micro-Transformer baseline vs ROGM (Resonant Orthogonal Gated Memory)
Both designed for ~50-70k params, vocab 16, d_model 64, 2 layers, CPU-friendly.
No O(N^2) in ROGM; Transformer uses standard O(N^2) attention for comparison.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

VOCAB_SIZE = 16

# ---------- Baseline: Micro Transformer ----------
class MicroTransformerBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, d_ff=128, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (B,L,D)
        B, L, D = x.shape
        residual = x
        x_norm = self.ln1(x)
        # Causal mask: (L,L) with -inf for upper triangle
        # For torch MultiheadAttention, attn_mask shape (L,L) with -inf masking
        attn_mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask, need_weights=False)
        x = residual + self.dropout(attn_out)
        # MLP
        residual = x
        x_norm = self.ln2(x)
        x = residual + self.dropout(self.mlp(x_norm))
        return x

class MicroTransformer(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=64, n_layers=2, n_heads=4, d_ff=128, max_len=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([MicroTransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        B, L = input_ids.shape
        assert L <= self.max_len, f"seq len {L} exceeds max_len {self.max_len}"
        pos = torch.arange(0, L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.token_embed(input_ids) + self.pos_embed(pos)
        # dropout?
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.head(x) # (B,L,V)
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

# ---------- Novel: ROGM ----------
class ROGMBlock(nn.Module):
    """
    Resonant Orthogonal Gated Memory Block
    - Input-dependent Householder reflection for orthogonal recurrence
    - Holographic matrix memory for associative recall
    - Oscillatory phase gating
    """
    def __init__(self, d_model=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.ln_pre = nn.LayerNorm(d_model)
        self.ln_post = nn.LayerNorm(d_model)
        
        # projections
        self.proj_q = nn.Linear(d_model, d_model, bias=False)
        self.proj_k = nn.Linear(d_model, d_model, bias=False)
        self.proj_v = nn.Linear(d_model, d_model, bias=False)
        self.proj_u = nn.Linear(d_model, d_model, bias=False) # householder vector
        self.proj_u2 = nn.Linear(d_model, d_model, bias=False) # second householder for full rotation (2 reflections = rotation)
        self.proj_gate = nn.Linear(d_model, d_model, bias=False)
        self.proj_phi = nn.Linear(d_model, d_model, bias=False) # phase
        # learned frequencies per dimension
        self.omega = nn.Parameter(torch.randn(d_model) * 0.2)
        self.memory_decay = nn.Parameter(torch.ones(d_model) * 0.0) # logits for sigmoid decay per dim, init 0 -> 0.5? But we use scalar 0.9 fixed for now; keep learnable later
        # output projection after memory+state fusion
        self.proj_out = nn.Linear(d_model, d_model, bias=False)
        # MLP - keep same as transformer (d_ff=128, i.e., 2*d_model) for fair param count
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.GELU(),
            nn.Linear(d_model*2, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B,L,D)
        Returns: (B,L,D) with residual & mlp incorporated
        Includes recurrent scan loop (sequential for correctness).
        """
        B, L, D = x.shape
        # pre-norm? We'll apply ln_pre per token before projections inside loop via alternative: apply to x once
        # But for recurrence, we want per-step input normalized: x_norm = ln_pre(x) then loop uses x_norm
        x_norm = self.ln_pre(x)
        # Precompute projections for all positions in parallel (for efficiency), then scan sequentially using precomputed tensors
        # This still respects causality because recurrence uses state, but projections are parallel.
        qs = self.proj_q(x_norm) # B,L,D
        ks = self.proj_k(x_norm)
        vs = self.proj_v(x_norm)
        us = self.proj_u(x_norm)
        us2 = self.proj_u2(x_norm)
        gates = torch.sigmoid(self.proj_gate(x_norm)) # B,L,D
        phis = torch.tanh(self.proj_phi(x_norm)) * math.pi # B,L,D phase in [-pi, pi]
        
        # Oscillator modulation: sin(omega * t + phi)
        # omega (D) broadcast with time step t
        # Create time vector shape (L,1) or (L,D)
        # Let's compute osc for each t: osc[t] = sin(omega * t + phi[:,t,:])
        # Using broadcasting
        # Initialize states
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype) # (B,D)
        # Holographic memory: M shape (B,D,D) -> matrix per sample
        M = torch.zeros(B, D, D, device=x.device, dtype=x.dtype)
        # Decay: use fixed 0.95 scalar for now; or learnable per-dim mean
        decay_val = 0.92  # could be sigmoid(self.memory_decay).mean()
        # For learnable per-dim decay, use per-dimension scaling:
        # decay_vec = torch.sigmoid(self.memory_decay) # D
        decay_vec = torch.sigmoid(self.memory_decay) # D
        # Use scalar average for stability of outer update; but we can apply per-row scaling later
        # We'll apply M decay as M * decay_scalar, plus optionally per dimension.
        # Precompute time-dependent oscillation matrix: we could compute per step inside loop.
        outputs = []
        # For efficiency, we could try to use torch.einsum for retrieval.
        # We'll loop over L (32 -> fine).
        for t in range(L):
            q_t = qs[:, t, :] # B,D
            k_t = ks[:, t, :]
            v_t = vs[:, t, :]
            u_t = us[:, t, :]
            u2_t = us2[:, t, :]
            g_t = gates[:, t, :] # B,D
            phi_t = phis[:, t, :] # B,D
            
            # --- Orthogonal recurrence via double Householder ---
            # First reflection using u_t
            # Normalize u
            u_norm = F.normalize(u_t, dim=-1, eps=1e-6) # B,D
            # projection of h onto u
            proj = (h * u_norm).sum(dim=-1, keepdim=True) # B,1
            h_ref = h - 2 * proj * u_norm # B,D
            
            # Second reflection using u2_t for full O(d) group (rotation)
            u2_norm = F.normalize(u2_t, dim=-1, eps=1e-6)
            proj2 = (h_ref * u2_norm).sum(dim=-1, keepdim=True)
            h_ref2 = h_ref - 2 * proj2 * u2_norm
            
            # Oscillatory gain: omega * t + phi_t
            # omega is D, so omega * t broadcasts
            # Need to handle per-sample phi_t
            osc = torch.sin(self.omega * t + phi_t)  # B,D (self.omega Bcast)
            # Modulate reflected state
            # Use osc as amplitude multiplier: (1 + 0.15*osc)
            h_osc = h_ref2 * (1.0 + 0.15 * osc)
            
            # Gated update: combine oscillatory reflected prior state with new input v_t
            # h_t = g * h_osc + (1-g) * tanh(v_t)
            # Use per-dim gate
            v_t_act = torch.tanh(v_t)
            h_new = g_t * h_osc + (1 - g_t) * v_t_act
            
            h = h_new
            
            # --- Holographic Memory update & retrieval ---
            # Retrieve before updating with current k,v (causal: use previous memory)
            # r = M^T q (or M @ q depending convention)
            # M shape B,D,D, q B,D => r B,D
            # Use einsum 'bde,bd->be' where M_{d,e} with q_d
            # Note: need to ensure order: M = sum_k k outer v, so M_{d,e}=k_d * v_e, retrieval M^T q gives sum_d M_{d,e} q_d? Wait dimensions: outer (k (D), v (D)) => matrix D x D index (d,e) . Contract with q_d over d yields e vector weighted by similarity q·k
            r_t = torch.einsum('bde,bd->be', M, q_t)  # B,D
            # Non-linear gating of retrieval: tanh + scale
            r_t = torch.tanh(r_t) * 0.5
            
            # Combine state + retrieval for this position
            y_t = h + r_t  # B,D
            outputs.append(y_t)
            
            # Update memory with current k_t, v_t (post-retrieval, for future steps)
            # kv outer product
            # Apply decay: M = M * decay_scalar + kv
            # For per-dim decay: we could scale M by decay_vec row and col: M * decay_vec[None,:,None] * decay_vec[None,None,:] ? Simpler use scalar per step for now, but also incorporate learnable decay_vec averaging effect
            # Use scalar decay 0.92 + small per-dim variance via adding 0.02 * decay_vec mean?
            # Let's use per-batch scalar decay = decay_vec.mean() * 0.1 + 0.85 => around 0.9
            # But to include per-dim nuance, we scale outer product contribution by (1 - decay)
            # Simpler: M = M * 0.9 + 0.1 * kv
            # We'll use adaptive decay: scalar = torch.sigmoid(self.memory_decay.mean()) * 0.2 + 0.8 => ~0.9
            # For efficiency, compute scalar once outside loop: we already have decay_vec, take mean
            # Use fixed for now: 0.92
            kv = torch.einsum('bd,be->bde', k_t, v_t) # B,D,D
            # Decay per element using outer decay? Use M * decay_vec broadcasting:
            # M_scaled = M * decay_vec[None,:,None]  # decay rows
            # Let's apply per-dimension decay on both axes: average
            # M = M * decay_vec.mean() + kv * (1 - decay_vec.mean()) * 0.5
            # Actually incorporate decay_vec: M = M * decay_vec[None,:,None] * decay_vec[None,None,:]? That would square. Keep simple.
            M = M * decay_val + kv * (1 - decay_val)  # weighted
            # Alternative use learnable decay_val = torch.sigmoid(self.memory_decay).mean()
            # For now fixed.
        # Stack outputs: B,L,D
        out_seq = torch.stack(outputs, dim=1)
        # Apply residual connection with input x (pre-ORGM)
        # We also have projected output per position
        out_seq = self.proj_out(out_seq)
        out_seq = self.dropout(out_seq)
        # Residual + post norm + MLP (Transformer style)
        residual = x + out_seq
        # Actually should have layer norm after adding? We'll use ln_post before MLP
        # For stability, we do: x_res = x + out_seq -> ln -> mlp -> residual
        h_mlp_in = self.ln_post(residual)
        mlp_out = self.mlp(h_mlp_in)
        mlp_out = self.dropout(mlp_out)
        out = residual + mlp_out
        return out

class ROGMModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=64, n_layers=2, max_len=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        # No positional embedding: relies on oscillatory time code (but we keep optional learned? disabled to prove no pos needed)
        # For fairness, we could keep no pos
        self.layers = nn.ModuleList([ROGMBlock(d_model, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Parameter):
            pass

    def forward(self, input_ids):
        B, L = input_ids.shape
        assert L <= self.max_len
        x = self.token_embed(input_ids) # B,L,D
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

def get_model(name="transformer", **kwargs):
    if name == "transformer":
        return MicroTransformer(**kwargs)
    elif name == "rogm":
        return ROGMModel(**kwargs)
    else:
        raise ValueError(name)

def model_stats():
    t = MicroTransformer()
    r = ROGMModel()
    print(f"Transformer params: {t.count_params()}")
    print(f"ROGM params: {r.count_params()}")
    # rough broken down
    for n,p in t.named_parameters():
        print(f"T {n} {p.shape} {p.numel()}")
    print("--- ROGM ---")
    for n,p in r.named_parameters():
        print(f"R {n} {p.shape} {p.numel()}")

if __name__ == "__main__":
    model_stats()
    # test forward
    B, L = 2, 8
    x = torch.randint(0, VOCAB_SIZE, (B, L))
    for name in ["transformer", "rogm"]:
        m = get_model(name, vocab_size=VOCAB_SIZE, d_model=64, n_layers=2, max_len=64)
        logits = m(x)
        print(f"{name} logits shape {logits.shape} sample logits {logits[0,0,:5].tolist()}")
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), torch.randint(0, VOCAB_SIZE, (B*L,)))
        loss.backward()
        print(f"  loss {loss.item():.4f} grad ok")
