# lottery-bias-kit

## Overview
`lottery-bias-kit` is a reusable Python library that packages the omegaXiv lottery-bias methodology into a clean API.
It exposes regime segmentation, replication-constrained FDR screening, partial-identification robustness, staged transfer checks, H5 reliability-constrained recalibration, and integrated scorecards as importable components.

## Installation
Canonical user installation flow:
1. `pip install omegaxiv`
2. `ox install lottery-bias-kit==0.1.0`

Maintainer/dev-only source installation (not the canonical end-user path):
- `pip install packages/lottery-bias-kit/dist/omegaxiv_lottery_bias_kit-0.1.0-py3-none-any.whl`

## Configuration
`LotteryBiasAnalyzer` accepts explicit seed lists and structured per-stage configuration.
Use `LotteryBiasAnalyzer.default_config()` as a baseline, then override stage parameters (`h1`, `h2`, `h3`, `h4`, `h5`) for your audit protocol.

## Usage Examples
```python
from lottery_bias_kit import LotteryBiasAnalyzer, sample_iid_draws

analyzer = LotteryBiasAnalyzer(seeds=[11, 23])
frame = sample_iid_draws(n_draws=320, seed=11)
result = analyzer.run_full(frame)

print(result["p4"]["composite_reproducibility_score"])
print(result["h4"]["brier_lift_vs_pooled"])
print(result["h5"]["summary"]["worst_case_regret_margin"])
```

Symbolic invariants:
```python
from lottery_bias_kit import compute_sympy_invariants

checks = compute_sympy_invariants()
print(checks["sign_identification_equivalence_cnf"])
```

## Troubleshooting
- If `temporal_split` fails on very small inputs, increase sample size beyond 250 rows.
- If optimization metrics are unstable across seeds, run more seeds and inspect `changepoint_consensus_rate`.
- If FDR monotonicity fails frequently, enable isotonic envelopes in `h2.isotonic_envelope`.
- If calibration diagnostics degrade, reduce transfer complexity by lowering `h4.eta_fused`.
- If H5 rank stability is noisy, increase `h5.bootstrap_reps` and inspect `bootstrap_top_rank_frequency`.
