# PHASE 1: Hypothesis Generation & Internet Validation

**Date:** 2026-09-17 UTC  
**Objective:** Conceptualize fundamentally new token routing / state-tracking mechanism that does not rely on O(N^2) self-attention and surpasses Transformer/Mamba/RWKV limitations in reasoning.

## 1. Baseline Literature Validation (Web Search)

### Query Results Summary:

**Mamba / SSM Limitations (validated 2025-2026 sources):**
- Mamba excels at byte-level modeling & resource-constrained edge, linear scalability, 70% memory reduction [1][zylos.ai]
- **Core weaknesses identified:**
  1. In-Context Learning (ICL): 15-point gap on MMLU vs Transformers (1.1T tokens) [1]
  2. Copying & Retrieval: Transformers with 10x params outperform Mamba; phonebook lookup remains challenging [1]
  3. Multi-Query Associative Recall (MQAR): degrades with longer inputs; stateful bottleneck [1]
  4. Root cause: **Fixed-size state** compressed sequentially → information loss, cannot scale state with input length [1][Survey Mamba 2408.01129]
  5. **State-tracking failure:** Merrill et al. ICML 2024 "Illusion of State" proves SSMs cannot solve permutation composition, chess moves, code evaluation, entity tracking beyond TC0. Expressiveness limited like Transformers despite recurrent formulation [2][proceedings.mlr.press/merrill24a]
  6. Sarrof et al. 2024: **Diagonal Mamba cannot solve parity (C2) in finite precision** for arbitrary lengths, even though parity is simplest solvable group [3]
  7. General finding: Mamba-1 uses diagonal A, Mamba-2 scalar*I → severely restricted; time-varying selection helps but loses convolution equivalence and still limited [4]

**RWKV Limitations:**
- Replaces quadratic QK with scalar linear attention; O(n) cost [RWKV arxiv 2305.13048]
- Limitation: funneling through single vector across time → cannot "look back" at previous tokens; learned time-decay mechanistically limited vs full attention
- Sensitive to prompt construction/position; desired information should be placed after question [RWKV]
- Larger internal states proposed as mitigation but fundamentally limited

**Hybrid & Alternatives (2025-2026):**
- Hybrids (6 attention + 58 SSD layers) outperform 64 pure SSD layers → pure SSM insufficient [1]
- NAtS-L (2026) token-level hybrid adaptive linear/softmax per token → bottleneck remains softmax layers
- Breaking Quadratic Barriers non-attention LLM (2025) → modular long-context but still not addressing state-tracking expressiveness
- ISANP etc. reduce query complexity but not reasoning

**Validated Gap:** No existing architecture solves **all three simultaneously** with O(N) or O(N log N):
1. Arbitrary parity / modular counting (finite-precision stable)
2. Non-solvable group permutation composition (S5, S3) requiring TC1
3. MQAR retrieval without fixed-state bottleneck
4. Without O(N^2) attention

→ Need **input-dependent orthogonal transitions** (not diagonal) + **expansive associative memory** (not fixed vector) + **oscillatory gating** (stable long-range propagation) in unified architecture.

Citations:
[1] Zylos Research 2026-01-21, [2] NeurIPS 2024 START, [3] Merrill ICML 2024 Illusion of State, [4] Survey Mamba 2408.01129, [5] RWKV 2305.13048, [6] NAtS-L 2602.03681

## 2. Novel Hypothesis: Resonant Orthogonal Gated Memory (ROGM)

### Name: **ROGM - Resonant Orthogonal Gated Memory** (alias: Holographic Oscillator with Adaptive Orthogonal Transitions - HOAOT)

**Core Mathematical Mutation (not found in literature):**

We mutate Mamba's selective SSM along **three orthogonal axes** that were individually hinted but never combined:

**A) Input-Dependent Orthogonal Recurrence (IDOR) via Householder Reflections:**
- Prior SSM: `h_t = diag(A_t) * h_{t-1} + B_t x_t` where A_t diagonal → provably cannot compose non-commutative groups (Merrill, Sarrof).
- **Mutation:** `A_t = H(u_t)` where `H(u) = I - 2 u u^T / ||u||^2` is an **input-dependent Householder reflection** (orthogonal, `H^T H = I`). Composition of 2 reflections yields any rotation in O(d) group. Stacking 2 blocks yields rotation composition depth.
- Why novel: Prior work suggested "input-dependent transition matrices" (Merrill's proposed fix) but only theorized dense matrices O(d^2) expensive. Householder parameterization achieves **input-dependent orthogonal** with O(d) compute and preserves norm (no vanishing/exploding, crucial for parity where gradient starvation kills diagonal Mamba). Orthogonal recurrence can represent any finite group element including S5 permutation matrix, thus escapes TC0.

**B) Holographic Associative Matrix Memory (HAMM) with Decayed Fast Weights:**
- Prior Mamba/RWKV: fixed vector state `h ∈ R^d` (capacity d).
- Mutation: maintain **matrix memory** `M_t ∈ R^{d×d}` updated via `M_t = λ M_{t-1} + k_t ⊗ v_t` (outer product, Kanerva holographic / linear attention fast-weight) and retrieval `r_t = M_{t-1}^T q_t`.
- Capacity is O(d^2) (4096 vs 64) without O(N^2) cost: per-step O(d^2) compute, O(N d^2) total linear in N. Decay λ (≈0.9) prevents catastrophic interference, learned per-layer via sigmoid.
- Unlike Transformers' O(N^2) pairwise attention, this stores superposed key-value bindings and retrieves via learned query, enabling **content-based copying / MQAR** without recomputing attention matrix. Addresses Mamba phonebook failure.

**C) Oscillatory Phase Gating (OPG) for Stable Long-Range Propagation:**
- Inspired by coupled oscillator models (Kuramoto) not used in discrete token prediction.
- Each dimension has learnable frequency `ω_i`, phase `φ_i = tanh(W_φ x_t)·π`, amplitude modulation `g_t = sin(ω·t + φ)`.
- Modulates Householder-reflected state: `h_t = g_t ⊙ h_reflected + (1-g_t) ⊙ tanh(v_t)` with input-dependent gate `γ_t = σ(W_g x_t)`.
- Provides **temporal addressing** without positional encodings; oscillations create stable periodic attractors for counting/parity (like sinusoidal positional encodings but dynamic and input-modulated). Solves Mamba's "selection mechanism accumulates domain-specific features" issue by binding time to autonomous oscillation rather than learned convolution kernel.

**Unified Block (O(N) linear, no attention):**
1. Input embedding + per-token LayerNorm
2. Parallel projections: `q,k,v,u,φ`
3. **Recurrent loop** over sequence (scan, O(N)):
   - `u_norm → H(u)` reflection of `h_{t-1}`
   - Oscillatory gain `sin(ωt+φ)` + gate `γ`
   - Update `h_t`
   - Update holographic memory `M_t` with `k_t ⊗ v_t`
   - Retrieve `r_t = M_{t-1} q_t`
   - `y_t = h_t + r_t` (state + retrieval fusion)
4. Residual + LayerNorm + MLP (like Transformer)
5. Stack 2 blocks (depth solves solvable groups via cascade).

**Complexity:** O(N · d^2) per layer linear in N (d=64 → negligible vs O(N^2·d) transformer). No O(N^2) attention; no positional encodings; no convolution.

**Why not copy Mamba/RWKV:**
- Not diagonal SSM, not linear attention scalar, not convolution-based S4. Orthogonal Householder + holographic matrix + oscillator triad is **unseen** in searched literature (no ArXiv hit for "Householder reflection SSM oscillator holographic memory").

**Expected Superiority:**
- **Parity (C2):** Orthogonal norm-preserving reflections ↔ parity is XOR = rotation by π; stable over arbitrary length where diagonal Mamba fails in finite precision.
- **S3/S5 composition:** Orthogonal matrices form group O(d) that contains permutation group representation → can implement arbitrary permutation composition.
- **MQAR/copying:** Matrix memory O(d^2) superposed storage → can store up to ~d key-value pairs (vs vector state 1) → near-attention retrieval without quadratic cost.
- **Memory:** Fixed d×d per sequence → predictable 16KB per layer vs Transformer KV-cache O(N·d) growing with length.

## 3. Experimental Plan (Micro-Scale Proof-of-Concept)

- **Hardware:** 2 CPU, 4GB RAM (simulated), no GPU. Use CPU torch, batch size 32, sequence length 16-64, d_model=64, 2 layers for both models (~60k params each).
- **Baselines:** Micro-Transformer (2 layers, 4 heads, learned pos embed, ≈62k params) trained on identical synthetic reasoning data.
- **Dataset Generator:** Synthetic reasoning-only, RAM <100MB:
  - Task A: Parity (prefix parity 0..L)
  - Task B: S3 permutation composition (6 elements)
  - Task C: MQAR 4-pair associative recall
  - Vocab 16 disjoint (0-1 parity, 2-7 S3, 8-11 keys, 12-15 values) + mask for loss.
  - Train: 16k sequences length 32 mixed task distribution 40/30/30; Val: 2k holdout same distribution + 2k **longer length 64** (length generalization test where SSM/Transformer degrade).
- **Metrics:** Loss curve, per-task accuracy, memory RSS, generalization gap (train vs long-val), convergence speed.
- **Iterations:** Phase 3: Train 30 epochs CPU; Phase 4: Compare; if ROGM underperforms → debug Householder norm or memory decay; if outperforms → harder task (length 128, noisier filler, S5 attempt).

## 4. Falsifiability
- If ROGM fails on parity: inspect Householder rank (need 2 reflections for rotation) → add second Householder.
- If MQAR fails: increase memory decay or add top-k sparsity.
- If OOM: reduce d to 32 or batch to 16.

---
**Next:** Phase 2 Implementation (PyTorch from scratch, synthetic dataset, fused loop).
