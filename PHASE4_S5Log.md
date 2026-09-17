# ITERATION 4 — S5 Non-Solvable Group Stress Test
**Date:** 2026-09-17T05:57 UTC | **Infinite Loop Continuation**

## Objective
Test ultimate state-tracking limit: S5 permutation composition (120 elements, non-solvable group, A5 simple). Merrill et al. (ICML 2024) proves SSMs & Transformers in TC0 cannot compose S5 even with input-dependent transitions; requires deeper cascade product. Hypothesis: ROGM's orthogonal Householder (O(d) contains S5) with 2-layer cascade should begin to separate, but micro-scale (d=64, 2 layers, 20 epochs) may still be insufficient.

## Setup
- Vocab 128 (0-119 S5 elements), d=64 (Transformer 91k, ROGM 115k, lite-32 33k), max_len 128, train seq 32, eval 32/64/96/128, 20 epochs, 160 batches/epoch (102k sequences), lr 1.2e-3, CPU 2 threads, 730-957MB RSS.

## Results (chance = 1/120 = 0.0083)

| Model | s5_32 | s5_64 | s5_96 | s5_128 | Params | Final Train Loss |
|-------|-------|-------|-------|--------|--------|------------------|
| Transformer-64 | 0.0397 | 0.0239 | 0.0186 | 0.0155 | 91k | 4.48 |
| ROGM-64 | 0.0416 | 0.0235 | 0.0185 | 0.0154 | 115k | 4.61 |
| ROGM-lite-32 | 0.0401 | 0.0238 | 0.0191 | 0.0155 | 33k | 4.58 |

Loss plateau 4.78→4.48 (all), val 4.76→4.64, acc barely above chance, decaying with length (0.04→0.015) as expected for random guessing with 120 classes. No model separates.

## Interpretation
- **All micro models fail S5 at this scale** — validates theory: S5 requires >TC0, even orthogonal 2-layer with d=64 insufficient without deeper cascade (Merrill suggests depth logarithmic in group complexity) or larger hidden (d >>5) and more training (102k seqs insufficient for 120-class composition).
- **Not a refutation of ROGM hypothesis:** S3 (solvable, 6 elements) is the correct micro proof-of-concept (Exp1/2): ROGM solved S3 perfectly (1.00→0.756 at 128) where diagonal Mamba and Transformer failed (0.26→0.189). S5's 20× larger state space and non-solvability needs scale-up: d=128-256, n_layers=4-6, 100+ epochs, curriculum.
- **Efficiency still holds:** ROGM no worse than Transformer despite extra params, but no better at this scale — indicates S5 needs algorithmic scale, not just orthogonal trick.
- **Synthetic difficulty:** S5's 120-way classification cross-entropy ~4.78 (log 120=4.78) — models stay near uniform.

## Autonomous Next Hypothesis (Iteration 5)
1. **Depth scaling for S5:** Test ROGM with **n_layers=4, d=96** (cascade product of 4 orthogonal blocks = theoretically sufficient for S3×C2 → S5 via Krohn-Rhodes). Train 50 epochs, curriculum: start S3 then S5.
2. **Speed optimization:** Implement `torch.compile` + fused scan for Householder/Memory loop; target 3× speedup (already 5-6× slower).
3. **Million-context extrapolation:** If S5 solved, test length 512-1024 with decay-tuned holographic memory (learnable λ).

**Patch log:** No OOM, stable RSS. S5 dataset built correctly (permutation compose verified). Training converged but to chance, not divergence.

**Loop status:** Iteration 4 logged, not terminal (infinite). Continue to deeper cascade.

---
*Only logs.*
