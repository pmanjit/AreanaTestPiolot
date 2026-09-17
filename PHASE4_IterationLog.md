# PHASE 3 → 4 : Autonomous Evaluation & Iteration Log
**Timestamp:** 2026-09-17T04:38 UTC (after Experiment 1)

## Experiment 1 Results Summary (30 epochs, 200 batches/epoch, seq_len 32)
- **Transformer baseline (73k params):**
  - Train loss 0.99, Val loss 0.995, Val acc 0.465
  - Parity 0.615, S3 0.265, MQAR 0.500
  - Long (64) Val acc 0.419, Parity_long 0.558, S3_long 0.210, MQAR_long 0.497
  - Mem 730MB, Time 93s
  - **Plateaued early, failed to learn state-tracking**: Parity barely above chance (0.615 vs 0.5 random), S3 at chance 0.265 vs 0.166, no length generalization (S3_long 0.21).

- **ROGM (101k params, 1.39× transformer, linear O(N) not O(N²)):**
  - Train loss 0.050, Val loss 0.043, Val acc 0.979 (+0.514 vs baseline)
  - Parity **1.00**, S3 **1.00**, MQAR 0.507 (parity with baseline)
  - Long (64) Val acc 0.899 (+0.48), Parity_long 0.921 (+0.364), S3_long 0.899 (+0.690), MQAR_long 0.463 (-0.035 within tolerance)
  - Mem 838MB, Time 610s (≈6.5× slower per epoch due to Python sequential scan, but still <14min total, fits 4GB)
  - **Convergence curve**: Loss dropped from 1.01 (epoch1) → 0.66 → 0.28 → 0.17 → 0.11 within 4 epochs, vs transformer 1.24 →1.10 →1.07 plateau. ROGM exhibits steeper reasoning/generalization slope.

**Metrics Evaluation:**
- Reasoning wins (parity, s3, parity_long, s3_long) = 4/4 (>2 required)
- MQAR not catastrophically worse: True (0.507 vs 0.5, -0.035 on long)
- Long generalization win: True (+0.48)
- => **SUCCESS**: Novel architecture surpasses baseline micro-transformer in inherent reasoning & generalization despite linear complexity.

**Bottleneck Analysis:**
- **Time bottleneck**: ROGM loop is pure Python `for t in range(L)` with per-step `einsum('bde,bd->be')` for M retrieval and `einsum('bd,be->bde')` for memory update (O(B·D²·L)=32·4096·32≈4M ops per batch, but Python overhead dominates). Transformer uses fused `nn.MultiheadAttention` (optimized C++). This is implementation bottleneck, not algorithmic; can be optimized via `torch.compile`, chunked scan, or `torch.vmap` but not critical for proof-of-concept.
- **Memory**: ROGM M matrix B·D·D = 32·64·64=131k floats ≈0.5MB per layer, negligible. RSS 838MB vs 730MB stable, far below 4GB limit.
- **Accuracy bottleneck**: MQAR_long slightly worse (-0.035) → holographic decay 0.92 may be too aggressive for long sequences (forgets earlier KV after 64 steps). Also ROGM's oscillatory phase may overfit short length and extrapolate less perfectly to long (parity_long 0.921 vs 1.0 short). Both are tunable.
- **Redundant params**: ROGM 101k vs 73k (+38% due to 2 Householder + 7 projections vs 4). Could strip to single Householder or smaller d_model and still dominate.

**Phase 4 Decision (Autonomous, per CommandToFollow):**
Since experiment **succeeded**, we do NOT stop. We must optimize further, strip redundant parameters, or test harder task. We choose **both**:
1. **Harder synthetic reasoning task**: length 64 train → length 96/128 zero-shot generalization, plus *selective* parity with ignore tokens (30% distractors) requiring content-based forgetting — a stress test where Mamba's selection previously failed (Sarrof). Also increase MQAR to 8 pairs + longer filler to stress memory capacity.
2. **Stripped optimization**: create ROGM-lite variants:
   - **ROGM-lite (d=32, ~28k params, <½ baseline)** to test if superior reasoning persists with fewer params than baseline (parameter efficiency).
   - **Ablation**: ROGM-no-oscillator and ROGM-single-Householder to identify redundant components.

**Iteration 2 Plan:**
- Generate harder dataset generator `generate_selective_parity` and longer sequences (train L=48, eval L=96/128).
- Train 3 models on identical harder data: Transformer-64 (73k), ROGM-64 (101k), ROGM-lite-32 (28k) for 25 epochs (≈15min).
- Log loss curves, per-task, length generalization gap, memory/time, parameter efficiency (acc per param).
- Hypothesis: ROGM-lite-32 will still outperform Transformer-64 despite <½ params, proving mechanism not just capacity; ROGM-64 will maintain >0.85 acc on length 128 selective tasks where Transformer collapses to chance.

Proceed to Phase 1 (re-mutated hypothesis for harder task) → Phase 2 implementation → Phase 3 testing.
