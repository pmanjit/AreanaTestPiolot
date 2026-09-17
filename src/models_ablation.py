"""
ROGM Ablation Variants — for iteration 3
Supports flags: use_double (second Householder), use_osc (oscillator), use_memory (holographic)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

VOCAB_SIZE = 16

class ROGMBlockAblation(nn.Module):
    def __init__(self, d_model=64, dropout=0.1, use_double=True, use_osc=True, use_memory=True):
        super().__init__()
        self.d_model = d_model
        self.use_double = use_double
        self.use_osc = use_osc
        self.use_memory = use_memory
        self.ln_pre = nn.LayerNorm(d_model)
        self.ln_post = nn.LayerNorm(d_model)
        # projections
        self.proj_q = nn.Linear(d_model, d_model, bias=False) if use_memory else None
        self.proj_k = nn.Linear(d_model, d_model, bias=False) if use_memory else None
        self.proj_v = nn.Linear(d_model, d_model, bias=False)
        self.proj_u = nn.Linear(d_model, d_model, bias=False)
        self.proj_u2 = nn.Linear(d_model, d_model, bias=False) if use_double else None
        self.proj_gate = nn.Linear(d_model, d_model, bias=False)
        self.proj_phi = nn.Linear(d_model, d_model, bias=False) if use_osc else None
        self.omega = nn.Parameter(torch.randn(d_model) * 0.2) if use_osc else None
        self.memory_decay = nn.Parameter(torch.ones(d_model) * 0.0) if use_memory else None
        self.proj_out = nn.Linear(d_model, d_model, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.GELU(),
            nn.Linear(d_model*2, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        x_norm = self.ln_pre(x)
        # Precompute projections conditionally
        if self.use_memory:
            qs = self.proj_q(x_norm)
            ks = self.proj_k(x_norm)
        else:
            qs = ks = None
        vs = self.proj_v(x_norm)
        us = self.proj_u(x_norm)
        if self.use_double:
            us2 = self.proj_u2(x_norm)
        else:
            us2 = None
        gates = torch.sigmoid(self.proj_gate(x_norm))
        if self.use_osc:
            phis = torch.tanh(self.proj_phi(x_norm)) * math.pi
        else:
            phis = None

        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        if self.use_memory:
            M = torch.zeros(B, D, D, device=x.device, dtype=x.dtype)
            decay_val = 0.92
        else:
            M = None
            decay_val = None

        outputs = []
        for t in range(L):
            if self.use_memory:
                q_t = qs[:, t, :]
                k_t = ks[:, t, :]
            v_t = vs[:, t, :]
            u_t = us[:, t, :]
            if self.use_double:
                u2_t = us2[:, t, :]
            g_t = gates[:, t, :]
            if self.use_osc:
                phi_t = phis[:, t, :]

            # Orthogonal recurrence
            u_norm = F.normalize(u_t, dim=-1, eps=1e-6)
            proj = (h * u_norm).sum(dim=-1, keepdim=True)
            h_ref = h - 2 * proj * u_norm
            if self.use_double:
                u2_norm = F.normalize(u2_t, dim=-1, eps=1e-6)
                proj2 = (h_ref * u2_norm).sum(dim=-1, keepdim=True)
                h_ref2 = h_ref - 2 * proj2 * u2_norm
            else:
                h_ref2 = h_ref

            if self.use_osc:
                osc = torch.sin(self.omega * t + phi_t)
                h_osc = h_ref2 * (1.0 + 0.15 * osc)
            else:
                h_osc = h_ref2

            v_t_act = torch.tanh(v_t)
            h_new = g_t * h_osc + (1 - g_t) * v_t_act
            h = h_new

            if self.use_memory:
                r_t = torch.einsum('bde,bd->be', M, q_t)
                r_t = torch.tanh(r_t) * 0.5
                y_t = h + r_t
                kv = torch.einsum('bd,be->bde', k_t, v_t)
                M = M * decay_val + kv * (1 - decay_val)
            else:
                y_t = h
            outputs.append(y_t)

        out_seq = torch.stack(outputs, dim=1)
        out_seq = self.proj_out(out_seq)
        out_seq = self.dropout(out_seq)
        residual = x + out_seq
        h_mlp_in = self.ln_post(residual)
        mlp_out = self.mlp(h_mlp_in)
        mlp_out = self.dropout(mlp_out)
        out = residual + mlp_out
        return out

class ROGMAblationModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=64, n_layers=2, max_len=128, dropout=0.1, use_double=True, use_osc=True, use_memory=True):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([ROGMBlockAblation(d_model, dropout, use_double, use_osc, use_memory) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.use_double = use_double
        self.use_osc = use_osc
        self.use_memory = use_memory
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
        assert L <= self.max_len
        x = self.token_embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

def get_ablation_model(variant="full", d_model=64, n_layers=2, max_len=128):
    """
    variant: full, singleH, noOsc, noMem, minimal (singleH+noOsc), lite (d=32 full)
    """
    if variant == "full":
        return ROGMAblationModel(d_model=d_model, n_layers=n_layers, max_len=max_len, use_double=True, use_osc=True, use_memory=True)
    elif variant == "singleH":
        return ROGMAblationModel(d_model=d_model, n_layers=n_layers, max_len=max_len, use_double=False, use_osc=True, use_memory=True)
    elif variant == "noOsc":
        return ROGMAblationModel(d_model=d_model, n_layers=n_layers, max_len=max_len, use_double=True, use_osc=False, use_memory=True)
    elif variant == "noMem":
        return ROGMAblationModel(d_model=d_model, n_layers=n_layers, max_len=max_len, use_double=True, use_osc=True, use_memory=False)
    elif variant == "minimal":
        return ROGMAblationModel(d_model=d_model, n_layers=n_layers, max_len=max_len, use_double=False, use_osc=False, use_memory=True)
    elif variant == "noMemNoOsc":
        return ROGMAblationModel(d_model=d_model, n_layers=n_layers, max_len=max_len, use_double=True, use_osc=False, use_memory=False)
    elif variant == "lite":
        return ROGMAblationModel(d_model=32, n_layers=n_layers, max_len=max_len, use_double=True, use_osc=True, use_memory=True)
    else:
        raise ValueError(variant)

if __name__ == "__main__":
    for v in ["full","singleH","noOsc","noMem","minimal","noMemNoOsc"]:
        m = get_ablation_model(v, d_model=64)
        print(f"{v:12s} params {m.count_params()}")
    m = get_ablation_model("lite", d_model=64)
    print(f"lite (d32) params {m.count_params()}")
    # test forward
    import torch.nn.functional as F
    x = torch.randint(0, 16, (2,16))
    for v in ["full","singleH","noOsc","noMem"]:
        m = get_ablation_model(v, d_model=64, max_len=64)
        logits = m(x)
        loss = F.cross_entropy(logits.view(-1,16), torch.randint(0,16,(32,)))
        loss.backward()
        print(f"{v} forward ok loss {loss.item():.3f}")
