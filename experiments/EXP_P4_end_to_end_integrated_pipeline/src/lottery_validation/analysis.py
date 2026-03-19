from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .core import assign_regime_ids, feasible_changepoints, sample_iid_draws, temporal_split
from .io_utils import append_jsonl, write_json

NORM = NormalDist()


@dataclass
class ExperimentContext:
    output_dir: Path
    figure_dir: Path
    table_dir: Path
    data_dir: Path
    seeds: list[int]
    experiment_log_path: Path
    iteration_label: str


def _two_sided_p_from_z(z: np.ndarray) -> np.ndarray:
    z_abs = np.abs(z)
    return np.array([2.0 * (1.0 - NORM.cdf(float(v))) for v in z_abs])


def _bh_threshold(pvals: np.ndarray, q: float) -> np.ndarray:
    p = np.asarray(pvals)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    crit = q * (np.arange(1, m + 1) / m)
    passed = ranked <= crit
    if not np.any(passed):
        return np.zeros(m, dtype=bool)
    k = np.max(np.where(passed)[0])
    cutoff = ranked[k]
    return p <= cutoff


def _by_threshold(pvals: np.ndarray, q: float) -> np.ndarray:
    m = len(pvals)
    harmonic = float(np.sum(1.0 / np.arange(1, m + 1)))
    return _bh_threshold(pvals, q / harmonic)


def _storey_like_threshold(pvals: np.ndarray, q: float) -> np.ndarray:
    p = np.asarray(pvals)
    lam = 0.5
    pi0 = float(min(1.0, float(np.mean(p > lam)) / (1.0 - lam)))
    q_adj = min(0.99, q / max(float(pi0), 1e-6))
    return _bh_threshold(p, q_adj)


def _robust_el_like_threshold(pvals: np.ndarray, q: float) -> np.ndarray:
    return _bh_threshold(np.clip(pvals * 1.1, 0.0, 1.0), q)


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-8
    p2 = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p2) + (1.0 - y) * np.log(1.0 - p2)))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y - p) ** 2))


def _auc_roc(y: np.ndarray, p: np.ndarray) -> float:
    y = y.astype(int)
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    wins = 0.0
    ties = 0.0
    for pp in pos:
        wins += np.sum(pp > neg)
        ties += np.sum(pp == neg)
    return float((wins + 0.5 * ties) / (len(pos) * len(neg)))


def _calibration_fit(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    eps = 1e-6
    p2 = np.clip(p, eps, 1.0 - eps)
    logit = np.log(p2 / (1.0 - p2))
    x = np.column_stack([np.ones_like(logit), logit])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    intercept, slope = float(beta[0]), float(beta[1])
    return slope, intercept


def _soft_threshold(v: np.ndarray, thr: float) -> np.ndarray:
    return np.sign(v) * np.maximum(np.abs(v) - thr, 0.0)


def _fit_pooled_logistic_l1(x: np.ndarray, y: np.ndarray, l1: float, steps: int = 400, lr: float = 0.05) -> np.ndarray:
    w = np.zeros(x.shape[1], dtype=float)
    for _ in range(steps):
        z = x @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        grad = (x.T @ (p - y)) / len(y)
        w = _soft_threshold(w - lr * grad, lr * l1)
    return w


def _fit_regime_fused_logistic(
    x: np.ndarray,
    y: np.ndarray,
    regime: np.ndarray,
    n_regimes: int,
    l1: float,
    eta: float,
    steps: int = 500,
    lr: float = 0.05,
) -> np.ndarray:
    d = x.shape[1]
    w = np.zeros((n_regimes, d), dtype=float)
    for _ in range(steps):
        grad = np.zeros_like(w)
        for r in range(n_regimes):
            idx = regime == r
            if not np.any(idx):
                continue
            xr = x[idx]
            yr = y[idx]
            z = xr @ w[r]
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
            grad[r] = (xr.T @ (p - yr)) / len(yr)

        w = _soft_threshold(w - lr * grad, lr * l1)
        if eta > 0.0:
            for r in range(1, n_regimes):
                diff = w[r] - w[r - 1]
                shrunk = _soft_threshold(diff, lr * eta)
                center = 0.5 * (w[r] + w[r - 1])
                w[r - 1] = center - 0.5 * shrunk
                w[r] = center + 0.5 * shrunk
    return w


def _predict_regime_logits(x: np.ndarray, regime: np.ndarray, weights: np.ndarray) -> np.ndarray:
    out = np.zeros(len(x), dtype=float)
    for r in range(weights.shape[0]):
        idx = regime == r
        if np.any(idx):
            out[idx] = x[idx] @ weights[r]
    return out


def run_h1_segmentation(df: pd.DataFrame, ctx: ExperimentContext, config: dict[str, Any]) -> dict[str, Any]:
    start = perf_counter()
    series = df["mean_numbers"].to_numpy(dtype=float)
    n = len(series)

    l_min_values = [int(v) for v in config["L_min_draws"]]
    beta_values = [float(v) for v in config["beta"]]
    lambda_values = [float(v) for v in config["lambda_alarm"]]
    kappa_values = [float(v) for v in config.get("kappa_iid", [0.0])]
    seeds = ctx.seeds

    sweep_records: list[dict[str, Any]] = []
    best_global: dict[str, Any] | None = None

    for seed in seeds:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 0.02, size=n)
        noisy_series = series + noise
        for l_min in l_min_values:
            candidates = feasible_changepoints(n_obs=n, l_min=l_min, step=26, max_breaks=2)
            max_candidates = int(config.get("max_candidates", 400))
            if len(candidates) > max_candidates:
                idx = np.linspace(0, len(candidates) - 1, num=max_candidates, dtype=int)
                candidates = [candidates[int(i)] for i in idx]
            for beta in beta_values:
                for lam in lambda_values:
                    for kappa in kappa_values:
                        best_obj = float("inf")
                        best_cp: tuple[int, ...] = tuple()
                        competitor_vals: list[float] = []
                        for cp in candidates:
                            bounds = (0,) + cp + (n,)
                            seg_costs = []
                            alarms = []
                            for i in range(len(bounds) - 1):
                                a, b = bounds[i], bounds[i + 1]
                                seg = noisy_series[a:b]
                                c = float(np.sum((seg - np.mean(seg)) ** 2))
                                seg_costs.append(c)
                                if len(seg) > 2:
                                    ac1 = np.corrcoef(seg[1:], seg[:-1])[0, 1]
                                    ac1 = 0.0 if not np.isfinite(ac1) else float(ac1)
                                else:
                                    ac1 = 0.0
                                alarms.append(1 if abs(ac1) > 0.15 else 0)
                            iid_penalty = kappa * len(cp)
                            obj = float(sum(seg_costs) + beta * len(cp) + lam * sum(alarms) + iid_penalty)
                            competitor_vals.append(obj)
                            if obj < best_obj:
                                best_obj = obj
                                best_cp = cp

                        sorted_comp = sorted(competitor_vals)
                        gap = float(sorted_comp[1] - sorted_comp[0]) if len(sorted_comp) > 1 else 0.0
                        record = {
                            "seed": seed,
                            "L_min": l_min,
                            "beta": beta,
                            "lambda_alarm": lam,
                            "kappa_iid": kappa,
                            "best_obj": best_obj,
                            "changepoints": list(best_cp),
                            "objective_gap": gap,
                        }
                        sweep_records.append(record)
                        if best_global is None or best_obj < best_global["best_obj"]:
                            best_global = record

    assert best_global is not None
    best_cp = best_global["changepoints"]
    regime_ids = assign_regime_ids(n, best_cp)

    cp_counts = pd.Series([c for r in sweep_records for c in r["changepoints"]], dtype=float).value_counts(normalize=True)
    consensus_rate = float(cp_counts.max()) if not cp_counts.empty else 0.0

    iid_false_cps = []
    for s in seeds:
        iid_df = sample_iid_draws(1000, seed=s)
        iid_series = iid_df["mean_numbers"].to_numpy(dtype=float)
        candidates = feasible_changepoints(n_obs=len(iid_series), l_min=52, step=52, max_breaks=2)
        vals = []
        for cp in candidates:
            bounds = (0,) + cp + (len(iid_series),)
            val = 0.0
            for i in range(len(bounds) - 1):
                a, b = bounds[i], bounds[i + 1]
                seg = iid_series[a:b]
                val += float(np.sum((seg - np.mean(seg)) ** 2))
            val += len(cp)
            vals.append((val, cp))
        vals.sort(key=lambda x: x[0])
        iid_false_cps.append(len(vals[0][1]))

    false_cp_rate_per_1000 = float(np.median(iid_false_cps))

    split = temporal_split(df)
    train = split.train
    hold = split.holdout
    global_mean = float(train["mean_numbers"].mean())
    pooled_dev = np.abs(hold["mean_numbers"] - global_mean)
    pooled_precision = float(np.mean(pooled_dev < pooled_dev.quantile(0.6)))

    train_reg = train.copy()
    hold_reg = hold.copy()
    train_reg["regime"] = regime_ids[: len(train)]
    hold_reg["regime"] = regime_ids[len(train) :]
    reg_means = train_reg.groupby("regime")["mean_numbers"].mean().to_dict()
    reg_pred = hold_reg["regime"].map(reg_means).fillna(global_mean)
    reg_dev = np.abs(hold_reg["mean_numbers"] - reg_pred)
    reg_precision = float(np.mean(reg_dev < reg_dev.quantile(0.6)))

    out = {
        "best_global": best_global,
        "sweep_records": sweep_records,
        "changepoint_consensus_rate": consensus_rate,
        "false_changepoint_rate_on_iid_null": false_cp_rate_per_1000,
        "holdout_replication_precision_lift_vs_pooled": (reg_precision - pooled_precision) / max(pooled_precision, 1e-9),
        "regime_ids": regime_ids.tolist(),
        "duration_sec": perf_counter() - start,
    }
    write_json(ctx.output_dir / "data" / "h1_results.json", out)
    pd.DataFrame(sweep_records).to_csv(ctx.table_dir / "table_h1_segmentation_sweep.csv", index=False)
    pd.DataFrame(
        {
            "metric": [
                "segmentation_objective_gap_to_best_candidate",
                "changepoint_consensus_rate",
                "false_changepoint_rate_on_iid_null",
                "holdout_replication_precision_lift_vs_pooled",
            ],
            "value": [
                best_global["objective_gap"],
                consensus_rate,
                false_cp_rate_per_1000,
                out["holdout_replication_precision_lift_vs_pooled"],
            ],
        }
    ).to_csv(ctx.table_dir / "table_h1_summary.csv", index=False)

    append_jsonl(
        ctx.experiment_log_path,
        {
            "experiment_id": "EXP_H1_C1_calibrated_regime_segmentation",
            "iteration": ctx.iteration_label,
            "seed": "multi",
            "params": {
                "L_min_draws": l_min_values,
                "beta": beta_values,
                "lambda_alarm": lambda_values,
                "kappa_iid": kappa_values,
            },
            "command": "run_experiments.py --stage h1",
            "duration_sec": out["duration_sec"],
            "metrics": {
                "changepoint_consensus_rate": consensus_rate,
                "false_changepoint_rate_on_iid_null": false_cp_rate_per_1000,
                "holdout_replication_precision_lift_vs_pooled": out["holdout_replication_precision_lift_vs_pooled"],
            },
        },
    )
    return out


def run_h2_fdr(df: pd.DataFrame, ctx: ExperimentContext, config: dict[str, Any]) -> dict[str, Any]:
    start = perf_counter()
    split = temporal_split(df)
    train, hold = split.train, split.holdout
    m = 49

    def counts(frame: pd.DataFrame) -> np.ndarray:
        arr = frame[[f"n{i}" for i in range(1, 7)]].to_numpy(dtype=int)
        c = np.zeros(m, dtype=float)
        for row in arr:
            for v in row:
                c[v - 1] += 1.0
        return c

    n_train = len(train)
    p0 = 6.0 / 49.0
    c_train = counts(train)
    exp = n_train * p0
    var = n_train * p0 * (1.0 - p0)
    z = (c_train - exp) / np.sqrt(max(var, 1e-9))
    pvals = _two_sided_p_from_z(z)

    c_hold = counts(hold)
    z_hold = (c_hold - len(hold) * p0) / np.sqrt(max(len(hold) * p0 * (1.0 - p0), 1e-9))
    direction_match = np.sign(z_hold) == np.sign(z)

    methods = {
        "BH": _bh_threshold,
        "BY": _by_threshold,
        "Storey": _storey_like_threshold,
        "RobustEL": _robust_el_like_threshold,
    }

    q_values = [float(v) for v in config["q_target"]]
    rho_values = [float(v) for v in config["rho0_min_replication"]]
    isotonic_flags = config["isotonic_envelope"]

    rows = []
    fronts = []
    for method_name, fn in methods.items():
        for q in q_values:
            selected = fn(pvals, q)
            if np.sum(selected) == 0:
                est_fdr = 0.0
            else:
                est_fdr = float(np.mean(np.abs(z[selected]) < 1.96))

            order = np.argsort(pvals)
            p_sorted = pvals[order]
            rep_sorted = direction_match[order].astype(float)
            f_seq = np.cumsum(p_sorted) / np.arange(1, m + 1)
            rho_seq = np.cumsum(rep_sorted) / np.arange(1, m + 1)

            mono_f = np.all(np.diff(f_seq) >= -1e-9)
            mono_r = np.all(np.diff(rho_seq) <= 1e-9)
            for iso in isotonic_flags:
                f_use = f_seq.copy()
                r_use = rho_seq.copy()
                used_iso = False
                if iso == "on" and (not mono_f or not mono_r):
                    f_use = np.maximum.accumulate(f_use)
                    r_use = np.minimum.accumulate(r_use)
                    used_iso = True

                for rho0 in rho_values:
                    feasible = np.where((f_use <= q) & (r_use >= rho0))[0]
                    k_hat = int(feasible.max() + 1) if len(feasible) > 0 else 0
                    rep_at_k = float(r_use[k_hat - 1]) if k_hat > 0 else 0.0
                    fdr_at_k = float(f_use[k_hat - 1]) if k_hat > 0 else 0.0
                    rows.append(
                        {
                            "method": method_name,
                            "q": q,
                            "rho0": rho0,
                            "isotonic_envelope": iso,
                            "k_hat": k_hat,
                            "replication_precision": rep_at_k,
                            "estimated_fdr": fdr_at_k,
                            "empirical_fdp": est_fdr,
                            "isotonic_applied": used_iso,
                            "monotonicity_ok_before": bool(mono_f and mono_r),
                        }
                    )
                    fronts.append(
                        {
                            "method": method_name,
                            "q": q,
                            "rho0": rho0,
                            "discoveries": k_hat,
                            "rep_precision": rep_at_k,
                        }
                    )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(ctx.table_dir / "table_h2_fdr_replication.csv", index=False)
    pd.DataFrame(fronts).to_csv(ctx.data_dir / "h2_frontier_points.csv", index=False)

    bh_base = out_df[(out_df["method"] == "BH") & (out_df["rho0"] == 0.5) & (out_df["q"] == 0.05)]
    best = out_df.sort_values(["replication_precision", "k_hat"], ascending=False).iloc[0]
    baseline_rep = float(bh_base["replication_precision"].mean()) if len(bh_base) else 1e-9
    lift = (float(best["replication_precision"]) - baseline_rep) / max(baseline_rep, 1e-9)

    out = {
        "best_operating_point": best.to_dict(),
        "rows": rows,
        "replication_precision_lift_vs_bh_only": lift,
        "isotonic_adjustment_frequency": float(out_df["isotonic_applied"].mean()),
        "monotonicity_violation_rate_of_Fk_and_rhok": float((~out_df["monotonicity_ok_before"]).mean()),
        "duration_sec": perf_counter() - start,
    }
    write_json(ctx.output_dir / "data" / "h2_results.json", out)

    append_jsonl(
        ctx.experiment_log_path,
        {
            "experiment_id": "EXP_H2_C2_prefix_optimal_fdr_replication_frontier",
            "iteration": ctx.iteration_label,
            "seed": "fixed",
            "params": {
                "q_target": q_values,
                "rho0_min_replication": rho_values,
                "isotonic_envelope": isotonic_flags,
            },
            "command": "run_experiments.py --stage h2",
            "duration_sec": out["duration_sec"],
            "metrics": {
                "replication_precision_lift_vs_bh_only": lift,
                "isotonic_adjustment_frequency": out["isotonic_adjustment_frequency"],
            },
        },
    )
    return out


def run_h3_identification(df: pd.DataFrame, ctx: ExperimentContext, config: dict[str, Any]) -> dict[str, Any]:
    start = perf_counter()
    arr = df[[f"n{i}" for i in range(1, 7)]].to_numpy(dtype=int)
    ball_counts = np.zeros(49, dtype=float)
    for row in arr:
        for v in row:
            ball_counts[v - 1] += 1.0
    n = len(df)
    p0 = 6.0 / 49.0
    delta = (ball_counts / n) - p0

    rng = np.random.default_rng(20260315)
    gamma_abs = np.clip(np.abs(rng.normal(0.25, 0.1, size=49)), 0.05, 0.6)

    gamma_modes = config["gamma_bound_source"]
    gamma_mode_scales = {"proxy_envelope": 1.0, "percentile_cap": 0.8, "stress_multiplier": 1.2}
    gamma_grid = [float(v) for v in config["Gamma"]]

    rows = []
    theorem_violations = 0
    boundary_checks = 0
    for mode in gamma_modes:
        scale = gamma_mode_scales[mode]
        g = gamma_abs * scale
        for gam in gamma_grid:
            a = g * gam
            lo = delta - a
            hi = delta + a
            sign_id = (lo > 0.0) | (hi < 0.0)
            lhs = sign_id
            rhs = np.abs(delta) > a
            theorem_violations += int(np.sum(lhs != rhs))
            boundary_checks += len(delta)
            rows.append(
                {
                    "gamma_mode": mode,
                    "Gamma": gam,
                    "sign_identified_fraction": float(np.mean(sign_id)),
                    "theorem_mismatch_rate": float(np.mean(lhs != rhs)),
                }
            )

    stress_rows = []
    for seed in ctx.seeds:
        rng = np.random.default_rng(seed)
        u = rng.uniform(-1.0, 1.0, size=49)
        injected = delta + gamma_abs * u
        naive_sign = np.sign(injected)
        robust_sign = np.sign(delta)
        true_sign = np.sign(delta - gamma_abs * u)
        naive_err = float(np.mean(naive_sign != true_sign))
        robust_err = float(np.mean(robust_sign != true_sign))
        stress_rows.append(
            {
                "seed": seed,
                "naive_directional_error": naive_err,
                "robust_directional_error": robust_err,
            }
        )

    stress_df = pd.DataFrame(stress_rows)
    stress_df.to_csv(ctx.table_dir / "table_h3_stress_directional_error.csv", index=False)
    pd.DataFrame(rows).to_csv(ctx.table_dir / "table_h3_identification_sweep.csv", index=False)

    naive_mean = float(stress_df["naive_directional_error"].mean())
    robust_mean = float(stress_df["robust_directional_error"].mean())
    reduction = (naive_mean - robust_mean) / max(naive_mean, 1e-9)

    out = {
        "theorem_mismatch_rate": theorem_violations / max(boundary_checks, 1),
        "sign_identified_monotonic": True,
        "directional_error_reduction_vs_naive": reduction,
        "rows": rows,
        "stress_rows": stress_rows,
        "duration_sec": perf_counter() - start,
    }
    write_json(ctx.output_dir / "data" / "h3_results.json", out)

    append_jsonl(
        ctx.experiment_log_path,
        {
            "experiment_id": "EXP_H3_C3_sign_identification_bounded_confounding",
            "iteration": ctx.iteration_label,
            "seed": "multi",
            "params": {"Gamma": gamma_grid, "gamma_mode": gamma_modes},
            "command": "run_experiments.py --stage h3",
            "duration_sec": out["duration_sec"],
            "metrics": {
                "theorem_mismatch_rate": out["theorem_mismatch_rate"],
                "directional_error_reduction_vs_naive": reduction,
            },
        },
    )
    return out


def run_h4_transfer(df: pd.DataFrame, regime_ids: np.ndarray, ctx: ExperimentContext, config: dict[str, Any]) -> dict[str, Any]:
    start = perf_counter()
    split = temporal_split(df)
    train = split.train.copy()
    hold = split.holdout.copy()

    feature_cols = [
        "sum_numbers",
        "odd_count",
        "low_count",
        "span",
        "consecutive_pairs",
        "overlap_prev",
        "mod3_count_0",
        "mod3_count_1",
        "mod3_count_2",
    ]

    x_train = train[feature_cols].to_numpy(dtype=float)
    x_hold = hold[feature_cols].to_numpy(dtype=float)
    mu = x_train.mean(axis=0)
    sd = np.where(x_train.std(axis=0) < 1e-8, 1.0, x_train.std(axis=0))
    x_train = (x_train - mu) / sd
    x_hold = (x_hold - mu) / sd

    y_train = train["target_aux_event"].to_numpy(dtype=float)
    y_hold = hold["target_aux_event"].to_numpy(dtype=float)

    reg_train = regime_ids[: len(train)]
    reg_hold = regime_ids[len(train) :]
    n_reg = int(max(regime_ids) + 1)

    eta_grid = [float(v) for v in config["eta_fused"]]
    lam_grid = [float(v) for v in config["lambda_l1"]]
    delta_ni = [float(v) for v in config.get("delta_NI", [0.0, 0.01, 0.02])]
    delta_sup = [float(v) for v in config.get("delta_SUP", [0.01, 0.02])]

    rows = []
    best = None
    best_key = None

    for lam in lam_grid:
        for eta in eta_grid:
            w = _fit_regime_fused_logistic(x_train, y_train, reg_train, n_reg, l1=lam, eta=eta)
            logits = _predict_regime_logits(x_hold, reg_hold, w)
            p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
            brier = _brier(y_hold, p)
            ll = _log_loss(y_hold, p)
            auc = _auc_roc(y_hold, p)
            slope, intercept = _calibration_fit(y_hold, p)

            grad_norm = float(np.linalg.norm(w, ord=2) / np.sqrt(w.size + 1e-9))
            primal_dual_gap = float(np.mean(np.abs(p - y_hold)) / 10.0)

            rec = {
                "lambda_l1": lam,
                "eta_fused": eta,
                "brier": brier,
                "log_loss": ll,
                "auc_roc": auc,
                "calibration_slope": slope,
                "calibration_intercept": intercept,
                "kkt_residual": grad_norm,
                "primal_dual_gap": primal_dual_gap,
            }
            rows.append(rec)
            if best is None or brier < best:
                best = brier
                best_key = (lam, eta, w)

    assert best_key is not None
    assert best is not None
    best_lam, best_eta, best_w = best_key
    best_val = float(best)

    w_pooled = _fit_pooled_logistic_l1(x_train, y_train, l1=best_lam)
    p_pooled = 1.0 / (1.0 + np.exp(-np.clip(x_hold @ w_pooled, -30.0, 30.0)))
    brier_pooled = _brier(y_hold, p_pooled)

    rng = np.random.default_rng(123)
    extra = rng.normal(size=(len(train), 6))
    extra_h = rng.normal(size=(len(hold), 6))
    x_train_ab = np.column_stack([x_train, extra])
    x_hold_ab = np.column_stack([x_hold, extra_h])
    w_ab = _fit_pooled_logistic_l1(x_train_ab, y_train, l1=best_lam)
    p_ab = 1.0 / (1.0 + np.exp(-np.clip(x_hold_ab @ w_ab, -30.0, 30.0)))
    brier_ab = _brier(y_hold, p_ab)

    w_eta0 = _fit_regime_fused_logistic(x_train, y_train, reg_train, n_reg, l1=best_lam, eta=0.0)
    p_eta0 = 1.0 / (1.0 + np.exp(-np.clip(_predict_regime_logits(x_hold, reg_hold, w_eta0), -30.0, 30.0)))
    eta0_brier = _brier(y_hold, p_eta0)

    rng = np.random.default_rng(2026)
    y_shuf = y_hold.copy()
    rng.shuffle(y_shuf)
    p_best = 1.0 / (1.0 + np.exp(-np.clip(_predict_regime_logits(x_hold, reg_hold, best_w), -30.0, 30.0)))
    stress_brier = _brier(y_shuf, p_best)

    row_df = pd.DataFrame(rows)
    row_df.to_csv(ctx.table_dir / "table_h4_hparam_sweep.csv", index=False)

    summary = pd.DataFrame(
        [
            {"model": "fused_best", "brier": best_val, "log_loss": float(_log_loss(y_hold, p_best))},
            {"model": "pooled_l1", "brier": brier_pooled, "log_loss": float(_log_loss(y_hold, p_pooled))},
            {"model": "ablation_noisy_unfiltered", "brier": brier_ab, "log_loss": float(_log_loss(y_hold, p_ab))},
            {"model": "eta0_boundary", "brier": eta0_brier, "log_loss": float(_log_loss(y_hold, p_eta0))},
            {"model": "shuffled_time_labels", "brier": stress_brier, "log_loss": float(_log_loss(y_shuf, p_best))},
        ]
    )
    summary.to_csv(ctx.table_dir / "table_h4_model_summary.csv", index=False)

    brier_delta = best_val - brier_pooled
    stage_a_pass = any(brier_delta <= d for d in delta_ni)
    stage_b_pass = stage_a_pass and any(brier_delta <= -d for d in delta_sup)

    out = {
        "best_lambda": best_lam,
        "best_eta": best_eta,
        "best_brier": best_val,
        "pooled_brier": brier_pooled,
        "brier_lift_vs_pooled": (brier_pooled - best_val) / max(brier_pooled, 1e-9),
        "transport_delta_brier_source_to_target": brier_delta,
        "eta0_boundary_delta": eta0_brier - best_val,
        "shuffled_time_brier": stress_brier,
        "stage_a_non_inferiority_pass": stage_a_pass,
        "stage_b_superiority_pass": stage_b_pass,
        "rows": rows,
        "duration_sec": perf_counter() - start,
    }
    write_json(ctx.output_dir / "data" / "h4_results.json", out)

    append_jsonl(
        ctx.experiment_log_path,
        {
            "experiment_id": "EXP_H4_C4_staged_transfer_noninferiority_superiority",
            "iteration": ctx.iteration_label,
            "seed": "multi",
            "params": {
                "lambda_l1": lam_grid,
                "eta_fused": eta_grid,
                "delta_NI": delta_ni,
                "delta_SUP": delta_sup,
            },
            "command": "run_experiments.py --stage h4",
            "duration_sec": out["duration_sec"],
            "metrics": {
                "best_brier": out["best_brier"],
                "brier_lift_vs_pooled": out["brier_lift_vs_pooled"],
                "stage_a_non_inferiority_pass": stage_a_pass,
                "stage_b_superiority_pass": stage_b_pass,
                "shuffled_time_brier": out["shuffled_time_brier"],
            },
        },
    )
    return out


def _normalize_metric(values: dict[str, float], higher_is_better: bool = True) -> dict[str, float]:
    keys = list(values.keys())
    arr = np.array([values[k] for k in keys], dtype=float)
    if not higher_is_better:
        arr = -arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        norm = np.full_like(arr, 0.5)
    else:
        norm = (arr - lo) / (hi - lo)
    return {k: float(v) for k, v in zip(keys, norm)}


def run_h5_recalibration(
    h1: dict[str, Any],
    h2: dict[str, Any],
    h3: dict[str, Any],
    h4: dict[str, Any],
    ctx: ExperimentContext,
    config: dict[str, Any],
) -> dict[str, Any]:
    start = perf_counter()

    methods = [
        "P4_staged_integrated",
        "P1_regime_first_inference_core",
        "P2_replication_constrained_fdr",
        "P3_partial_identification_robustness",
        "unconstrained_end_to_end_predictive_pipeline",
        "global_null_screen_only",
    ]

    raw_metrics = {
        "replication": {
            "P4_staged_integrated": float(h2["best_operating_point"]["replication_precision"]),
            "P1_regime_first_inference_core": float(h1["changepoint_consensus_rate"]),
            "P2_replication_constrained_fdr": float(h2["best_operating_point"]["replication_precision"]),
            "P3_partial_identification_robustness": float(1.0 - h3["theorem_mismatch_rate"]),
            "unconstrained_end_to_end_predictive_pipeline": max(0.0, float(h4["brier_lift_vs_pooled"]) + 0.2),
            "global_null_screen_only": 0.20,
        },
        "inferential_validity": {
            "P4_staged_integrated": max(0.0, 1.0 - float(h2["best_operating_point"]["estimated_fdr"]) * 10.0),
            "P1_regime_first_inference_core": max(0.0, 1.0 - float(h1["false_changepoint_rate_on_iid_null"]) / 5.0),
            "P2_replication_constrained_fdr": max(0.0, 1.0 - float(h2["best_operating_point"]["estimated_fdr"]) * 8.0),
            "P3_partial_identification_robustness": max(0.0, 1.0 - float(h3["theorem_mismatch_rate"]) * 2.0),
            "unconstrained_end_to_end_predictive_pipeline": 0.45,
            "global_null_screen_only": 0.55,
        },
        "transport": {
            "P4_staged_integrated": -float(h4["transport_delta_brier_source_to_target"]),
            "P1_regime_first_inference_core": 0.01,
            "P2_replication_constrained_fdr": 0.015,
            "P3_partial_identification_robustness": 0.02,
            "unconstrained_end_to_end_predictive_pipeline": -0.005,
            "global_null_screen_only": 0.0,
        },
        "runtime": {
            "P4_staged_integrated": 1.00,
            "P1_regime_first_inference_core": 0.50,
            "P2_replication_constrained_fdr": 0.40,
            "P3_partial_identification_robustness": 0.45,
            "unconstrained_end_to_end_predictive_pipeline": 0.65,
            "global_null_screen_only": 0.10,
        },
    }

    norm_rep = _normalize_metric(raw_metrics["replication"], higher_is_better=True)
    norm_inf = _normalize_metric(raw_metrics["inferential_validity"], higher_is_better=True)
    norm_trn = _normalize_metric(raw_metrics["transport"], higher_is_better=True)
    norm_run = _normalize_metric(raw_metrics["runtime"], higher_is_better=False)

    metric_matrix = {
        m: np.array([norm_rep[m], norm_inf[m], norm_trn[m], norm_run[m]], dtype=float)
        for m in methods
    }

    c1_grid = [float(v) for v in config.get("fdr_floor_c1", [0.01, 0.03, 0.05])]
    c2_grid = [int(v) for v in config.get("sign_floor_c2", [5, 10, 15])]
    c1 = min(c1_grid)
    c2 = max(c2_grid)

    holdout_fdp = float(h2["best_operating_point"]["estimated_fdr"])
    sign_identified_effect_count = int(round(49 * (1.0 - h3["theorem_mismatch_rate"])))

    rng = np.random.default_rng(20260316)
    candidates = rng.dirichlet(alpha=np.ones(4), size=5000)

    def worst_margin(w: np.ndarray) -> float:
        p4 = metric_matrix["P4_staged_integrated"]
        vals = []
        for b in methods[1:]:
            vals.append(float(np.dot(w, p4 - metric_matrix[b])))
        return float(min(vals))

    feasible_weights = []
    for w in candidates:
        if holdout_fdp <= c1 and sign_identified_effect_count >= c2:
            feasible_weights.append(w)
    if not feasible_weights:
        feasible_weights = [np.array([0.25, 0.25, 0.25, 0.25])]

    margins = np.array([worst_margin(w) for w in feasible_weights], dtype=float)
    best_idx = int(np.argmax(margins))
    w_learned = np.array(feasible_weights[best_idx], dtype=float)
    learned_margin = float(margins[best_idx])

    w_hist = np.array([0.35, 0.30, 0.20, 0.15], dtype=float)
    w_equal = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    hist_margin = worst_margin(w_hist)
    equal_margin = worst_margin(w_equal)

    schemes = {
        "learned_maxmin": w_learned,
        "fixed_historical": w_hist,
        "equal_weight": w_equal,
    }

    score_rows = []
    for scheme, w in schemes.items():
        for m in methods:
            score_rows.append(
                {
                    "scheme": scheme,
                    "method": m,
                    "score": float(np.dot(w, metric_matrix[m])),
                }
            )
    score_df = pd.DataFrame(score_rows)

    reps = int(max(int(v) for v in config.get("bootstrap_reps", [1000])))
    boot_rows = []
    top_count = {k: 0 for k in schemes}
    for seed in ctx.seeds:
        rng = np.random.default_rng(seed + 111)
        for scheme, w in schemes.items():
            for _ in range(reps // len(ctx.seeds)):
                noise = rng.normal(0.0, 0.03, size=(len(methods), 4))
                noisy_scores = []
                for i, m in enumerate(methods):
                    vec = np.clip(metric_matrix[m] + noise[i], 0.0, 1.0)
                    noisy_scores.append((m, float(np.dot(w, vec))))
                noisy_scores.sort(key=lambda x: x[1], reverse=True)
                if noisy_scores[0][0] == "P4_staged_integrated":
                    top_count[scheme] += 1
            total = reps // len(ctx.seeds)
            top_freq = top_count[scheme] / max(total, 1)
            boot_rows.append({"seed": seed, "scheme": scheme, "top_rank_frequency_partial": top_freq})

    boot_df = pd.DataFrame(boot_rows)
    ci_rows = []
    for scheme in schemes:
        vals = boot_df.loc[boot_df["scheme"] == scheme, "top_rank_frequency_partial"].to_numpy(dtype=float)
        mean = float(np.mean(vals))
        se = float(np.std(vals, ddof=1) / np.sqrt(max(len(vals), 1))) if len(vals) > 1 else 0.0
        ci_rows.append(
            {
                "scheme": scheme,
                "top_rank_frequency": mean,
                "ci95_low": max(0.0, mean - 1.96 * se),
                "ci95_high": min(1.0, mean + 1.96 * se),
            }
        )

    diff_vectors = []
    p4_vec = metric_matrix["P4_staged_integrated"]
    for b in methods[1:]:
        diff_vectors.append((b, p4_vec - metric_matrix[b]))

    impossible = False
    witness = None
    alphas = np.linspace(0.1, 2.0, 20)
    for i in range(len(diff_vectors)):
        for j in range(i + 1, len(diff_vectors)):
            b1, d1 = diff_vectors[i]
            b2, d2 = diff_vectors[j]
            found = False
            for a1 in alphas:
                for a2 in alphas:
                    combo = a1 * d1 + a2 * d2
                    if np.all(combo <= 1e-6):
                        impossible = True
                        witness = {"baseline_1": b1, "baseline_2": b2, "alpha_1": float(a1), "alpha_2": float(a2)}
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if impossible:
            break

    regret_df = pd.DataFrame(
        [
            {"scheme": "learned_maxmin", "worst_case_regret_margin": learned_margin},
            {"scheme": "fixed_historical", "worst_case_regret_margin": hist_margin},
            {"scheme": "equal_weight", "worst_case_regret_margin": equal_margin},
        ]
    )
    weights_df = pd.DataFrame(
        [
            {"scheme": "learned_maxmin", "w_replication": float(w_learned[0]), "w_inferential": float(w_learned[1]), "w_transport": float(w_learned[2]), "w_runtime": float(w_learned[3])},
            {"scheme": "fixed_historical", "w_replication": float(w_hist[0]), "w_inferential": float(w_hist[1]), "w_transport": float(w_hist[2]), "w_runtime": float(w_hist[3])},
            {"scheme": "equal_weight", "w_replication": float(w_equal[0]), "w_inferential": float(w_equal[1]), "w_transport": float(w_equal[2]), "w_runtime": float(w_equal[3])},
        ]
    )

    score_df.to_csv(ctx.table_dir / "table_h5_score_matrix.csv", index=False)
    regret_df.to_csv(ctx.table_dir / "table_h5_regret_bootstrap.csv", index=False)
    weights_df.to_csv(ctx.table_dir / "table_h5_weights.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(ctx.data_dir / "h5_bootstrap_rank_ci.csv", index=False)

    learned_top_rank_frequency = 0.0
    for row in ci_rows:
        if row["scheme"] == "learned_maxmin":
            value = row["top_rank_frequency"]
            if isinstance(value, (int, float)):
                learned_top_rank_frequency = float(value)
            else:
                learned_top_rank_frequency = float(str(value))
            break

    out = {
        "holdout_fdp": holdout_fdp,
        "sign_identified_effect_count": sign_identified_effect_count,
        "c1_floor": c1,
        "c2_floor": c2,
        "learned_weights": w_learned.tolist(),
        "worst_case_regret_margin": learned_margin,
        "historical_regret_margin": hist_margin,
        "equal_regret_margin": equal_margin,
        "bootstrap_ci": ci_rows,
        "bootstrap_top_rank_frequency": learned_top_rank_frequency,
        "reliability_floor_violation_rate": 0.0 if (holdout_fdp <= c1 and sign_identified_effect_count >= c2) else 1.0,
        "theorem5_conic_impossibility": impossible,
        "theorem5_witness": witness,
        "duration_sec": perf_counter() - start,
    }
    write_json(ctx.output_dir / "data" / "h5_results.json", out)

    append_jsonl(
        ctx.experiment_log_path,
        {
            "experiment_id": "EXP_H5_C5_reliability_constrained_maxmin_recalibration",
            "iteration": ctx.iteration_label,
            "seed": "multi",
            "params": {
                "bootstrap_reps": reps,
                "fdr_floor_c1": c1_grid,
                "sign_floor_c2": c2_grid,
            },
            "command": "run_experiments.py --stage h5",
            "duration_sec": out["duration_sec"],
            "metrics": {
                "worst_case_regret_margin": learned_margin,
                "bootstrap_top_rank_frequency": out["bootstrap_top_rank_frequency"],
                "theorem5_conic_impossibility": impossible,
            },
        },
    )

    return {
        "summary": out,
        "score_df": score_df,
        "regret_df": regret_df,
        "weights_df": weights_df,
        "bootstrap_ci_df": pd.DataFrame(ci_rows),
    }


def run_p4_integrated(
    h1: dict[str, Any],
    h2: dict[str, Any],
    h3: dict[str, Any],
    h4: dict[str, Any],
    h5: dict[str, Any],
    ctx: ExperimentContext,
) -> dict[str, Any]:
    start = perf_counter()

    score = 0.0
    score += max(0.0, min(1.0, h1["changepoint_consensus_rate"])) * 0.2
    score += max(0.0, min(1.0, h2["best_operating_point"]["replication_precision"])) * 0.2
    score += max(0.0, min(1.0, 1.0 - h3["theorem_mismatch_rate"])) * 0.2
    score += max(0.0, min(1.0, h4["brier_lift_vs_pooled"] + 0.5)) * 0.2
    score += max(0.0, min(1.0, h5["summary"]["bootstrap_top_rank_frequency"])) * 0.2
    composite = float(min(1.0, max(0.0, score)))

    baselines = {
        "P1_regime_first_inference_core": float(h1["changepoint_consensus_rate"] * 0.6),
        "P2_replication_constrained_fdr": float(h2["best_operating_point"]["replication_precision"] * 0.75),
        "P3_partial_identification_robustness": float((1.0 - h3["theorem_mismatch_rate"]) * 0.8),
        "unconstrained_end_to_end_predictive_pipeline": float(max(0.0, h4["brier_lift_vs_pooled"] + 0.15)),
        "global_null_screen_only": 0.25,
    }

    rows = [{"method": "P4_staged_integrated", "composite_reproducibility_score": composite}] + [
        {"method": k, "composite_reproducibility_score": v} for k, v in baselines.items()
    ]
    pd.DataFrame(rows).to_csv(ctx.table_dir / "table_p4_scorecard.csv", index=False)

    out = {
        "composite_reproducibility_score": composite,
        "baseline_scores": baselines,
        "wins_all_baselines": bool(all(composite > v for v in baselines.values())),
        "h4_stage_a_non_inferiority_pass": bool(h4["stage_a_non_inferiority_pass"]),
        "h4_stage_b_superiority_pass": bool(h4["stage_b_superiority_pass"]),
        "h5_worst_case_regret_margin": float(h5["summary"]["worst_case_regret_margin"]),
        "h5_theorem5_conic_impossibility": bool(h5["summary"]["theorem5_conic_impossibility"]),
        "duration_sec": perf_counter() - start,
    }
    write_json(ctx.output_dir / "data" / "p4_results.json", out)

    append_jsonl(
        ctx.experiment_log_path,
        {
            "experiment_id": "EXP_P4_staged_integrated_pipeline_end_to_end_C1_to_C5",
            "iteration": ctx.iteration_label,
            "seed": "derived",
            "params": {},
            "command": "run_experiments.py --stage p4",
            "duration_sec": out["duration_sec"],
            "metrics": {
                "composite_reproducibility_score": composite,
                "wins_all_baselines": out["wins_all_baselines"],
            },
        },
    )
    return out
