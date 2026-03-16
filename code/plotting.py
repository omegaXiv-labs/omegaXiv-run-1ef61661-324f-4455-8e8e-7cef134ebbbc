from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def apply_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def plot_h1_h2(h1: dict[str, Any], h2_df: pd.DataFrame, out_path: Path) -> None:
    apply_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    sweep = pd.DataFrame(h1["sweep_records"])
    if len(sweep):
        sns.lineplot(data=sweep, x="beta", y="best_obj", hue="L_min", ax=axes[0, 0], estimator="mean", errorbar=("ci", 95))
    axes[0, 0].set_xlabel("beta penalty")
    axes[0, 0].set_ylabel("Objective value")
    axes[0, 0].set_title("H1 Objective vs Penalty")
    axes[0, 0].legend(title="L_min", loc="best")

    cp_counts = pd.Series([c for r in h1["sweep_records"] for c in r["changepoints"]], dtype=float).value_counts().sort_index()
    if len(cp_counts):
        axes[0, 1].plot(cp_counts.index, cp_counts.values, marker="o", label="CP selection count")
    axes[0, 1].set_xlabel("Changepoint index (draw)")
    axes[0, 1].set_ylabel("Selection count")
    axes[0, 1].set_title("H1 Changepoint Consensus")
    axes[0, 1].legend(loc="best")

    if len(h2_df):
        sns.lineplot(data=h2_df, x="rho0", y="k_hat", hue="method", style="q", ax=axes[1, 0], errorbar=None)
    axes[1, 0].set_xlabel("rho0 replication threshold")
    axes[1, 0].set_ylabel("Discoveries (k_hat)")
    axes[1, 0].set_title("H2 Stability Frontier")
    axes[1, 0].legend(loc="best")

    if len(h2_df):
        sns.lineplot(data=h2_df, x="q", y="estimated_fdr", hue="method", ax=axes[1, 1], errorbar=("ci", 95))
    axes[1, 1].set_xlabel("Target q")
    axes[1, 1].set_ylabel("Estimated FDR")
    axes[1, 1].set_title("H2 FDR Behavior")
    axes[1, 1].legend(loc="best")

    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def plot_h3(h3_df: pd.DataFrame, stress_df: pd.DataFrame, out_path: Path) -> None:
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    sns.lineplot(data=h3_df, x="Gamma", y="sign_identified_fraction", hue="gamma_mode", ax=axes[0], errorbar=("ci", 95))
    axes[0].set_xlabel("Gamma confounding budget")
    axes[0].set_ylabel("Sign-identified fraction")
    axes[0].set_title("H3 Identification Survival")
    axes[0].legend(loc="best")

    tmp = stress_df.melt(id_vars=["seed"], value_vars=["naive_directional_error", "robust_directional_error"], var_name="method", value_name="directional_error")
    sns.barplot(data=tmp, x="method", y="directional_error", ax=axes[1], errorbar=("ci", 95))
    axes[1].set_xlabel("Method")
    axes[1].set_ylabel("Directional error")
    axes[1].set_title("H3 Stress Error")
    axes[1].legend(["95% CI"], loc="upper right")

    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def plot_h4(h4_sweep_df: pd.DataFrame, h4_summary_df: pd.DataFrame, out_path: Path) -> None:
    apply_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    sns.lineplot(data=h4_sweep_df, x="eta_fused", y="brier", hue="lambda_l1", ax=axes[0, 0], errorbar=("ci", 95))
    axes[0, 0].set_xlabel("eta fused penalty")
    axes[0, 0].set_ylabel("Brier score")
    axes[0, 0].set_title("H4 Brier by Hyperparameters")
    axes[0, 0].legend(loc="best")

    sns.scatterplot(data=h4_sweep_df, x="calibration_slope", y="calibration_intercept", hue="lambda_l1", style="eta_fused", ax=axes[0, 1])
    axes[0, 1].set_xlabel("Calibration slope")
    axes[0, 1].set_ylabel("Calibration intercept")
    axes[0, 1].set_title("H4 Calibration Diagnostics")
    axes[0, 1].legend(loc="best")

    sns.barplot(data=h4_summary_df, x="model", y="brier", ax=axes[1, 0], errorbar=("ci", 95))
    axes[1, 0].set_xlabel("Model")
    axes[1, 0].set_ylabel("Brier score")
    axes[1, 0].set_title("H4 Baseline Comparison")
    axes[1, 0].tick_params(axis="x", rotation=20)

    sns.barplot(data=h4_summary_df, x="model", y="log_loss", ax=axes[1, 1], errorbar=("ci", 95))
    axes[1, 1].set_xlabel("Model")
    axes[1, 1].set_ylabel("Log-loss")
    axes[1, 1].set_title("H4 Log-loss Comparison")
    axes[1, 1].tick_params(axis="x", rotation=20)

    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def plot_p4(score_df: pd.DataFrame, bootstrap_ci_df: pd.DataFrame, weights_df: pd.DataFrame, regret_df: pd.DataFrame, out_path: Path) -> None:
    apply_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    p4_scores = score_df[score_df["scheme"] == "learned_maxmin"]
    sns.barplot(data=p4_scores, x="method", y="score", ax=axes[0, 0], errorbar=("ci", 95))
    axes[0, 0].set_xlabel("Pipeline")
    axes[0, 0].set_ylabel("Composite score")
    axes[0, 0].set_title("P4/H5 Learned-Weight Scores")
    axes[0, 0].tick_params(axis="x", rotation=20)

    sns.barplot(data=regret_df, x="scheme", y="worst_case_regret_margin", ax=axes[0, 1], errorbar=("ci", 95))
    axes[0, 1].axhline(0.0, color="black", linewidth=1.0, linestyle="--", label="Zero margin")
    axes[0, 1].set_xlabel("Weight scheme")
    axes[0, 1].set_ylabel("Worst-case regret margin")
    axes[0, 1].set_title("H5 Max-Min Margin")
    axes[0, 1].legend(loc="best")

    weights_long = weights_df.melt(id_vars=["scheme"], var_name="component", value_name="weight")
    sns.barplot(data=weights_long, x="component", y="weight", hue="scheme", ax=axes[1, 0], errorbar=("ci", 95))
    axes[1, 0].set_xlabel("Metric component")
    axes[1, 0].set_ylabel("Weight")
    axes[1, 0].set_title("H5 Weight Allocation")
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 0].legend(loc="best")

    axes[1, 1].errorbar(
        x=np.arange(len(bootstrap_ci_df)),
        y=bootstrap_ci_df["top_rank_frequency"].to_numpy(dtype=float),
        yerr=np.vstack(
            [
                bootstrap_ci_df["top_rank_frequency"] - bootstrap_ci_df["ci95_low"],
                bootstrap_ci_df["ci95_high"] - bootstrap_ci_df["top_rank_frequency"],
            ]
        ).astype(float),
        fmt="o",
        capsize=4,
        label="Top-rank freq (95% CI)",
    )
    axes[1, 1].set_xticks(np.arange(len(bootstrap_ci_df)))
    axes[1, 1].set_xticklabels(bootstrap_ci_df["scheme"].tolist(), rotation=20)
    axes[1, 1].set_xlabel("Weight scheme")
    axes[1, 1].set_ylabel("Top-rank frequency")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_title("Bootstrap Ranking Stability")
    axes[1, 1].legend(loc="best")

    fig.savefig(out_path, format="pdf")
    plt.close(fig)
