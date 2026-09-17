# AreanaTestPiolot — Resonant Orthogonal Gated Memory (ROGM)

Autonomous AI Research Scientist — micro-scale proof-of-concept for novel neural architecture surpassing Transformer limitations.

- [x] Start
- [x] Succeed — ROGM exhibits superior reasoning/generalization curves vs baseline micro-transformer
- [ ] Failed? — No; see REPORT.md

## Quick Results
- **Exp1 (seq 32):** ROGM 0.979 val acc vs Transformer 0.465 (+0.514), long 64 0.899 vs 0.419 (+0.48), parity 1.00 vs 0.61, S3 1.00 vs 0.27
- **Exp2 Harder (train 48 → eval 128, selective parity):** ROGM-lite-32 (26k params, <½ Transformer) 0.921 overall at 128 vs Transformer 0.436, selective parity 1.00 vs 0.537
- **Constraints:** 4GB RAM, 2 CPU, no GPU, CPU torch, <1GB RSS, O(N) not O(N²)

## Artifacts
- `REPORT.md` — full technical logs (Phase 1-4, search validation, mutation, loss curves)
- `PHASE1_Hypothesis.md` — hypothesis & internet validation
- `PHASE4_IterationLog.md` — iteration evaluation
- `src/` — PyTorch models & synthetic dataset generators
- `logs/` — JSON metrics, training logs, plots
  - `experiment_comparison.png` — 4-panel loss/accuracy/per-task/generalization
  - `comparison_*.json` — reproducible metrics

## Reproduce
```bash
python3 src/dataset.py
python3 src/models.py
python3 train.py
python3 train_harder.py
python3 plot_results.py
```

Branch: `arena/01a0ad97-areanatestpiolot` — continuous autonomous loop, no human interaction.
