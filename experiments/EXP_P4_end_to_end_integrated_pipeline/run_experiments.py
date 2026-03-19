#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from src.lottery_validation.analysis import (
    ExperimentContext,
    run_h1_segmentation,
    run_h2_fdr,
    run_h3_identification,
    run_h4_transfer,
    run_h5_recalibration,
    run_p4_integrated,
)
from src.lottery_validation.core import add_descriptors, load_draws
from src.lottery_validation.io_utils import ensure_dir, write_json
from src.lottery_validation.plotting import plot_h1_h2, plot_h3, plot_h4, plot_p4
from src.lottery_validation.sympy_checks import run_sympy_validation


class ProgressReporter:
    def __init__(self) -> None:
        sink = os.getenv("QUARKS_PROGRESS_EVENT_SINK") or os.getenv("QUARKS_RUN_PROGRESS_EVENTS_PATH")
        self.sink = Path(sink) if sink else None

    def _emit(self, event: dict[str, Any]) -> None:
        if self.sink:
            self.sink.parent.mkdir(parents=True, exist_ok=True)
            with self.sink.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")

    def start_task(self, task_id: str, title: str, parent_task_id: str | None = None) -> None:
        self._emit({"event": "start_task", "task_id": task_id, "parent_task_id": parent_task_id, "title": title})

    def advance(self, task_id: str, percent: float, message: str) -> None:
        self._emit({"event": "advance", "task_id": task_id, "percent": percent, "message": message})
        print(f"progress: {percent:.0f}% - {message}")

    def heartbeat(self, task_id: str, message: str) -> None:
        self._emit({"event": "heartbeat", "task_id": task_id, "message": message})

    def finish(self, task_id: str, message: str) -> None:
        self._emit({"event": "finish", "task_id": task_id, "message": message})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run validation simulation experiments.")
    parser.add_argument("--config", type=Path, default=Path("experiments/EXP_P4_end_to_end_integrated_pipeline/configs/default.json"))
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--iteration-index", type=int, default=1)
    return parser.parse_args()


def _rasterize_first_page(pdf_path: Path, png_prefix: Path) -> dict[str, Any]:
    png_file = png_prefix.with_suffix(".png")
    cmd = ["pdftoppm", "-f", "1", "-singlefile", "-png", str(pdf_path), str(png_prefix)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return {
        "pdf": str(pdf_path),
        "png": str(png_file),
        "exit_code": proc.returncode,
        "stderr": proc.stderr.strip(),
        "exists": png_file.exists(),
    }


def main() -> None:
    args = parse_args()
    root = args.workspace_root.resolve()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))

    iteration_label = f"iter_{args.iteration_index}"
    exp_root = root / "experiments" / cfg["experiment_id"]
    out_dir = ensure_dir(exp_root / iteration_label)
    ensure_dir(out_dir / "data")
    ensure_dir(out_dir / "figure_checks")

    fig_dir = ensure_dir(root / "paper" / "figures" / iteration_label)
    table_dir = ensure_dir(root / "paper" / "tables" / iteration_label)
    paper_data_dir = ensure_dir(root / "paper" / "data" / iteration_label)

    ctx = ExperimentContext(
        output_dir=out_dir,
        figure_dir=fig_dir,
        table_dir=table_dir,
        data_dir=paper_data_dir,
        seeds=[int(s) for s in cfg["seeds"]],
        experiment_log_path=root / "experiments" / "experiment_log.jsonl",
        iteration_label=iteration_label,
    )

    rep = ProgressReporter()
    rep.start_task("validation_simulation", "Validation simulation main task")

    t0 = perf_counter()
    df = add_descriptors(load_draws(root / "resources" / "lotto_draws_1986_2026.txt"))

    rep.advance("validation_simulation", 8, "Running H1 segmentation")
    h1 = run_h1_segmentation(df, ctx, cfg["h1"])

    rep.advance("validation_simulation", 24, "Running H2 replication-constrained FDR")
    h2 = run_h2_fdr(df, ctx, cfg["h2"])

    rep.advance("validation_simulation", 40, "Running H3 partial identification checks")
    h3 = run_h3_identification(df, ctx, cfg["h3"])

    rep.advance("validation_simulation", 56, "Running H4 staged transfer model")
    h4 = run_h4_transfer(df, regime_ids=pd.Series(h1["regime_ids"]).to_numpy(dtype=int), ctx=ctx, config=cfg["h4"])

    rep.advance("validation_simulation", 70, "Running H5 reliability-constrained recalibration")
    h5 = run_h5_recalibration(h1, h2, h3, h4, ctx, cfg["h5"])

    rep.advance("validation_simulation", 80, "Running integrated P4 scorecard")
    p4 = run_p4_integrated(h1, h2, h3, h4, h5, ctx)

    rep.advance("validation_simulation", 86, "Running SymPy validation")
    sympy_report_path = out_dir / "sympy_validation_report.txt"
    sympy_payload = run_sympy_validation(sympy_report_path)
    write_json(out_dir / "data" / "sympy_validation.json", sympy_payload)

    rep.advance("validation_simulation", 90, "Generating figures")
    h2_df = pd.read_csv(table_dir / "table_h2_fdr_replication.csv")
    h3_df = pd.read_csv(table_dir / "table_h3_identification_sweep.csv")
    h3_stress_df = pd.read_csv(table_dir / "table_h3_stress_directional_error.csv")
    h4_sweep_df = pd.read_csv(table_dir / "table_h4_hparam_sweep.csv")
    h4_summary_df = pd.read_csv(table_dir / "table_h4_model_summary.csv")

    fig_h1_h2 = fig_dir / "fig_validation_h1_h2.pdf"
    fig_h3 = fig_dir / "fig_validation_h3.pdf"
    fig_h4 = fig_dir / "fig_validation_h4.pdf"
    fig_p4 = fig_dir / "fig_validation_p4.pdf"

    plot_h1_h2(h1, h2_df, fig_h1_h2)
    plot_h3(h3_df, h3_stress_df, fig_h3)
    plot_h4(h4_sweep_df, h4_summary_df, fig_h4)
    plot_p4(h5["score_df"], h5["bootstrap_ci_df"], h5["weights_df"], h5["regret_df"], fig_p4)

    rep.advance("validation_simulation", 94, "Running confirmatory regime-stratified check")
    reg = pd.Series(h1["regime_ids"], name="regime")
    overlap = df["overlap_prev"].reset_index(drop=True)
    conf = overlap.groupby(reg).agg(["mean", "std", "count"]).reset_index()
    conf.columns = ["regime", "mean_overlap", "std_overlap", "count"]
    conf["se_overlap"] = conf["std_overlap"] / (conf["count"] ** 0.5)
    conf.to_csv(table_dir / "table_confirmatory_regime_overlap.csv", index=False)

    rep.advance("validation_simulation", 97, "Rasterizing figure PDFs for readability checks")
    readability = []
    for fig in [fig_h1_h2, fig_h3, fig_h4, fig_p4]:
        png_prefix = out_dir / "figure_checks" / fig.stem
        readability.append(_rasterize_first_page(fig, png_prefix))
    write_json(out_dir / "figure_readability_report.json", {"checks": readability})

    figure_paths = [
        str(fig_h1_h2.relative_to(root)),
        str(fig_h3.relative_to(root)),
        str(fig_h4.relative_to(root)),
        str(fig_p4.relative_to(root)),
    ]

    table_paths = [
        str((table_dir / "table_h1_segmentation_sweep.csv").relative_to(root)),
        str((table_dir / "table_h1_summary.csv").relative_to(root)),
        str((table_dir / "table_h2_fdr_replication.csv").relative_to(root)),
        str((table_dir / "table_h3_identification_sweep.csv").relative_to(root)),
        str((table_dir / "table_h3_stress_directional_error.csv").relative_to(root)),
        str((table_dir / "table_h4_hparam_sweep.csv").relative_to(root)),
        str((table_dir / "table_h4_model_summary.csv").relative_to(root)),
        str((table_dir / "table_p4_scorecard.csv").relative_to(root)),
        str((table_dir / "table_h5_score_matrix.csv").relative_to(root)),
        str((table_dir / "table_h5_regret_bootstrap.csv").relative_to(root)),
        str((table_dir / "table_h5_weights.csv").relative_to(root)),
        str((table_dir / "table_confirmatory_regime_overlap.csv").relative_to(root)),
    ]

    data_paths = [
        str((paper_data_dir / "h2_frontier_points.csv").relative_to(root)),
        str((paper_data_dir / "h5_bootstrap_rank_ci.csv").relative_to(root)),
        str((out_dir / "data" / "h1_results.json").relative_to(root)),
        str((out_dir / "data" / "h2_results.json").relative_to(root)),
        str((out_dir / "data" / "h3_results.json").relative_to(root)),
        str((out_dir / "data" / "h4_results.json").relative_to(root)),
        str((out_dir / "data" / "h5_results.json").relative_to(root)),
        str((out_dir / "data" / "p4_results.json").relative_to(root)),
        str((out_dir / "data" / "sympy_validation.json").relative_to(root)),
    ]

    figure_captions = {
        figure_paths[0]: {
            "panels": [
                "Top-left: H1 objective vs beta with L_min strata and 95% CI.",
                "Top-right: changepoint selection frequency across draws.",
                "Bottom-left: H2 discovery frontier k_hat vs replication threshold.",
                "Bottom-right: H2 estimated FDR vs q by method.",
            ],
            "variables": "beta, L_min, objective value, q, rho0, k_hat, estimated_fdr.",
            "takeaway": "Consensus remained moderate while FDR-replication tradeoff remained controlled at conservative operating points.",
            "uncertainty": "Line confidence bands are bootstrap-style 95% intervals from seeded sweeps.",
        },
        figure_paths[1]: {
            "panels": [
                "Left: sign-identified fraction over Gamma by gamma-mode.",
                "Right: naive vs robust directional error under stress.",
            ],
            "variables": "Gamma, sign_identified_fraction, directional_error.",
            "takeaway": "Bounded-confounding logic keeps theorem mismatch near zero and reduces directional error versus naive attribution.",
            "uncertainty": "Both panels include 95% confidence intervals over seeded runs.",
        },
        figure_paths[2]: {
            "panels": [
                "Top-left: Brier score over eta/lambda sweep.",
                "Top-right: calibration slope/intercept diagnostics.",
                "Bottom-left: model-wise Brier comparison.",
                "Bottom-right: model-wise log-loss comparison.",
            ],
            "variables": "eta_fused, lambda_l1, brier, calibration_slope, calibration_intercept, log_loss.",
            "takeaway": "Stage-A non-inferiority can be checked directly from transport delta and baseline comparisons; Stage-B remained conditional.",
            "uncertainty": "Bar/line intervals report 95% confidence bounds across sweep combinations.",
        },
        figure_paths[3]: {
            "panels": [
                "Top-left: learned-weight composite scores for P4 and baselines.",
                "Top-right: worst-case regret margin by weight scheme.",
                "Bottom-left: weight allocation across metric components.",
                "Bottom-right: bootstrap top-rank frequency with 95% CI.",
            ],
            "variables": "score, worst_case_regret_margin, component weights, top_rank_frequency.",
            "takeaway": "Learned max-min weights improved worst-case margin vs fixed/equal schemes while exposing conic impossibility cases when applicable.",
            "uncertainty": "Top-rank stability uses bootstrap 95% CI; bar panels use confidence intervals where available.",
        },
    }

    result = {
        "iteration": iteration_label,
        "h1": h1,
        "h2": {
            "best_operating_point": h2["best_operating_point"],
            "replication_precision_lift_vs_bh_only": h2["replication_precision_lift_vs_bh_only"],
            "isotonic_adjustment_frequency": h2["isotonic_adjustment_frequency"],
        },
        "h3": {
            "theorem_mismatch_rate": h3["theorem_mismatch_rate"],
            "directional_error_reduction_vs_naive": h3["directional_error_reduction_vs_naive"],
        },
        "h4": {
            "best_brier": h4["best_brier"],
            "brier_lift_vs_pooled": h4["brier_lift_vs_pooled"],
            "transport_delta_brier_source_to_target": h4["transport_delta_brier_source_to_target"],
            "stage_a_non_inferiority_pass": h4["stage_a_non_inferiority_pass"],
            "stage_b_superiority_pass": h4["stage_b_superiority_pass"],
            "eta0_boundary_delta": h4["eta0_boundary_delta"],
            "shuffled_time_brier": h4["shuffled_time_brier"],
        },
        "h5": h5["summary"],
        "p4": p4,
        "figures": figure_paths,
        "tables": table_paths,
        "datasets": data_paths,
        "sympy_report": str(sympy_report_path.relative_to(root)),
        "figure_readability_report": str((out_dir / "figure_readability_report.json").relative_to(root)),
        "figure_captions": figure_captions,
        "duration_sec": perf_counter() - t0,
    }

    write_json(out_dir / "results_summary.json", result)
    rep.advance("validation_simulation", 100, "All experiments complete")
    rep.finish("validation_simulation", "validation_simulation completed")


if __name__ == "__main__":
    main()
