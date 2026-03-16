# Validation Simulation Experiments

This package executes the `validation_simulation` phase for the selected integrated path:
`EXP_P4_end_to_end_integrated_pipeline`.

## Goal

Implement and execute the experiment matrix from `phase_outputs/experiment_design.json` to validate hypotheses H1-H5 and the integrated staged P4 pipeline using reproducible code, figures, tables, and symbolic checks.

## Fixed Compute Envelope

The run profile used for manuscript evidence is fixed to:

- No GPU
- `<= 28` CPU-hours for a full staged run
- `<= 8` GB peak RAM
- `<= 250` MB optional external downloads (disabled by default for the reported run)

This keeps reruns comparable and aligned with the manuscript protocol section.

## Structure

- `run_experiments.py`: Thin CLI entrypoint for orchestration.
- `src/lottery_validation/core.py`: Data loading, descriptors, split helpers, changepoint utilities.
- `src/lottery_validation/analysis.py`: H1-H5 + integrated P4 experiment implementations.
- `src/lottery_validation/plotting.py`: Seaborn-styled multi-panel PDF figure generation.
- `src/lottery_validation/sympy_checks.py`: Symbolic theorem/identity validation checks.
- `configs/default.json`: Seeds and sweep parameter config.
- `tests/test_pipeline.py`: Reproducibility and iter-path output checks.

## Commands

Run from workspace root:

```bash
experiments/.venv/bin/python experiments/EXP_P4_end_to_end_integrated_pipeline/run_experiments.py --workspace-root . --iteration-index 1
experiments/.venv/bin/python -m ruff check experiments/EXP_P4_end_to_end_integrated_pipeline
experiments/.venv/bin/python -m mypy --config-file experiments/EXP_P4_end_to_end_integrated_pipeline/configs/mypy.ini experiments/EXP_P4_end_to_end_integrated_pipeline/src
experiments/.venv/bin/python -m pytest experiments/EXP_P4_end_to_end_integrated_pipeline/tests -q
```

## Outputs

- Experiment outputs: `experiments/EXP_P4_end_to_end_integrated_pipeline/iter_1/`
- Log file: `experiments/experiment_log.jsonl`
- Figures: `paper/figures/iter_1/*.pdf`
- Tables: `paper/tables/iter_1/*.csv`
- Data exports: `paper/data/iter_1/*.csv` and experiment JSON summaries.

## Notes

- The pipeline uses the local historical dataset `resources/lotto_draws_1986_2026.txt`.
- The symbolic report is generated at `experiments/EXP_P4_end_to_end_integrated_pipeline/iter_1/sympy_validation_report.txt`.
- Plot files contain labels/titles/legends only; manuscript-caption handoff notes are emitted in `iter_1/results_summary.json` under `figure_captions`.
- Discovery diagnostics must report isotonic-adjustment frequency as a first-class metric in addition to FDR/FDP and replication precision.
- Staged interpretation is mandatory in reporting: non-inferiority is required before any superiority claim, and integrated dominance claims are blocked when reliability floors or worst-case margins fail.
