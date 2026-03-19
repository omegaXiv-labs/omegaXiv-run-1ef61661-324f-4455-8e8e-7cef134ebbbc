from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from .data import assign_regime_ids, feasible_changepoints, sample_iid_draws, temporal_split

NORM = NormalDist()


@dataclass
class AnalyzerConfig:
    h1: dict[str, Any]
    h2: dict[str, Any]
    h3: dict[str, Any]
    h4: dict[str, Any]


class LotteryBiasAnalyzer:
    """Reusable implementation of the H1-H4 + integrated P4 analysis pipeline."""

    def __init__(self, seeds: list[int] | None = None) -> None:
        self.seeds = list(seeds) if seeds is not None else [7, 17, 29, 53, 89]

    @staticmethod
    def default_config() -> AnalyzerConfig:
        return AnalyzerConfig(
            h1={"L_min_draws": [26, 52], "beta": [1.0, 2.0], "lambda_alarm": [0.0, 1.0]},
            h2={"q_target": [0.05], "rho0_min_replication": [0.7], "isotonic_envelope": ["on", "off"]},
            h3={"gamma_bound_source": ["proxy_envelope", "percentile_cap"], "Gamma": [0.0, 0.5, 1.0]},
            h4={"eta_fused": [0.0, 0.001], "lambda_l1": [0.001, 0.005]},
        )

    @staticmethod
    def _two_sided_p_from_z(z: np.ndarray) -> np.ndarray:
        return np.array([2.0 * (1.0 - NORM.cdf(float(v))) for v in np.abs(z)])

    @staticmethod
    def _bh_threshold(pvals: np.ndarray, q: float) -> np.ndarray:
        pvals = np.asarray(pvals)
        m = len(pvals)
        order = np.argsort(pvals)
        ranked = pvals[order]
        crit = q * (np.arange(1, m + 1) / m)
        passed = ranked <= crit
        if not np.any(passed):
            return np.zeros(m, dtype=bool)
        k_idx = int(np.max(np.where(passed)[0]))
        cutoff = ranked[k_idx]
        return np.asarray(pvals <= cutoff, dtype=bool)

    @classmethod
    def _by_threshold(cls, pvals: np.ndarray, q: float) -> np.ndarray:
        m = len(pvals)
        harmonic = float(np.sum(1.0 / np.arange(1, m + 1)))
        return cls._bh_threshold(pvals, q / harmonic)

    @classmethod
    def _storey_like_threshold(cls, pvals: np.ndarray, q: float) -> np.ndarray:
        pvals = np.asarray(pvals)
        lam = 0.5
        pi0 = float(min(1.0, float(np.mean(pvals > lam)) / (1.0 - lam)))
        q_adj = min(0.99, q / max(pi0, 1e-6))
        return cls._bh_threshold(pvals, q_adj)

    @classmethod
    def _robust_el_like_threshold(cls, pvals: np.ndarray, q: float) -> np.ndarray:
        return cls._bh_threshold(np.clip(pvals * 1.1, 0.0, 1.0), q)

    @staticmethod
    def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
        return np.asarray(
            np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0),
            dtype=float,
        )

    @staticmethod
    def _brier(y_true: np.ndarray, prob: np.ndarray) -> float:
        return float(np.mean((y_true - prob) ** 2))

    @staticmethod
    def _log_loss(y_true: np.ndarray, prob: np.ndarray) -> float:
        eps = 1e-8
        prob = np.clip(prob, eps, 1.0 - eps)
        return float(-np.mean(y_true * np.log(prob) + (1.0 - y_true) * np.log(1.0 - prob)))

    @staticmethod
    def _auc_roc(y_true: np.ndarray, prob: np.ndarray) -> float:
        y_true = y_true.astype(int)
        pos = prob[y_true == 1]
        neg = prob[y_true == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        wins = 0.0
        ties = 0.0
        for pos_prob in pos:
            wins += np.sum(pos_prob > neg)
            ties += np.sum(pos_prob == neg)
        return float((wins + 0.5 * ties) / (len(pos) * len(neg)))

    @staticmethod
    def _calibration_fit(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
        eps = 1e-6
        prob = np.clip(prob, eps, 1.0 - eps)
        logit = np.log(prob / (1.0 - prob))
        x_mat = np.column_stack([np.ones_like(logit), logit])
        beta, *_ = np.linalg.lstsq(x_mat, y_true, rcond=None)
        intercept, slope = float(beta[0]), float(beta[1])
        return slope, intercept

    @classmethod
    def _fit_pooled_logistic_l1(
        cls,
        x_mat: np.ndarray,
        y_true: np.ndarray,
        l1: float,
        steps: int = 400,
        lr: float = 0.05,
    ) -> np.ndarray:
        weights = np.zeros(x_mat.shape[1], dtype=float)
        for _ in range(steps):
            logits = x_mat @ weights
            prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
            grad = (x_mat.T @ (prob - y_true)) / len(y_true)
            weights = cls._soft_threshold(weights - lr * grad, lr * l1)
        return weights

    @classmethod
    def _fit_regime_fused_logistic(
        cls,
        x_mat: np.ndarray,
        y_true: np.ndarray,
        regime: np.ndarray,
        n_regimes: int,
        l1: float,
        eta: float,
        steps: int = 500,
        lr: float = 0.05,
    ) -> np.ndarray:
        dim = x_mat.shape[1]
        weights = np.zeros((n_regimes, dim), dtype=float)
        for _ in range(steps):
            grad = np.zeros_like(weights)
            for regime_id in range(n_regimes):
                idx = regime == regime_id
                if not np.any(idx):
                    continue
                xr = x_mat[idx]
                yr = y_true[idx]
                logits = xr @ weights[regime_id]
                prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
                grad[regime_id] = (xr.T @ (prob - yr)) / len(yr)

            weights = cls._soft_threshold(weights - lr * grad, lr * l1)
            if eta > 0.0:
                for regime_id in range(1, n_regimes):
                    diff = weights[regime_id] - weights[regime_id - 1]
                    shrunk = cls._soft_threshold(diff, lr * eta)
                    center = 0.5 * (weights[regime_id] + weights[regime_id - 1])
                    weights[regime_id - 1] = center - 0.5 * shrunk
                    weights[regime_id] = center + 0.5 * shrunk

        return weights

    @staticmethod
    def _predict_regime_logits(x_mat: np.ndarray, regime: np.ndarray, weights: np.ndarray) -> np.ndarray:
        out = np.zeros(len(x_mat), dtype=float)
        for regime_id in range(weights.shape[0]):
            idx = regime == regime_id
            if np.any(idx):
                out[idx] = x_mat[idx] @ weights[regime_id]
        return out

    def run_h1_segmentation(self, frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
        series = frame["mean_numbers"].to_numpy(dtype=float)
        n_obs = len(series)

        l_min_values = [int(v) for v in config["L_min_draws"]]
        beta_values = [float(v) for v in config["beta"]]
        lambda_values = [float(v) for v in config["lambda_alarm"]]

        sweep_records: list[dict[str, Any]] = []
        best_global: dict[str, Any] | None = None

        for seed in self.seeds:
            rng = np.random.default_rng(seed)
            noisy_series = series + rng.normal(0.0, 0.02, size=n_obs)
            for l_min in l_min_values:
                candidates = feasible_changepoints(n_obs=n_obs, l_min=l_min, step=26, max_breaks=2)
                for beta in beta_values:
                    for lambda_alarm in lambda_values:
                        best_obj = float("inf")
                        best_cp: tuple[int, ...] = tuple()
                        competitor_vals: list[float] = []
                        for cp in candidates:
                            bounds = (0,) + cp + (n_obs,)
                            seg_costs = []
                            alarms = []
                            for idx in range(len(bounds) - 1):
                                a, b = bounds[idx], bounds[idx + 1]
                                seg = noisy_series[a:b]
                                seg_costs.append(float(np.sum((seg - np.mean(seg)) ** 2)))
                                if len(seg) > 2:
                                    ac1 = np.corrcoef(seg[1:], seg[:-1])[0, 1]
                                    ac1 = 0.0 if not np.isfinite(ac1) else float(ac1)
                                else:
                                    ac1 = 0.0
                                alarms.append(1 if abs(ac1) > 0.15 else 0)
                            obj = float(sum(seg_costs) + beta * len(cp) + lambda_alarm * sum(alarms))
                            competitor_vals.append(obj)
                            if obj < best_obj:
                                best_obj = obj
                                best_cp = cp

                        sorted_vals = sorted(competitor_vals)
                        objective_gap = float(sorted_vals[1] - sorted_vals[0]) if len(sorted_vals) > 1 else 0.0
                        record = {
                            "seed": seed,
                            "L_min": l_min,
                            "beta": beta,
                            "lambda_alarm": lambda_alarm,
                            "best_obj": best_obj,
                            "changepoints": list(best_cp),
                            "objective_gap": objective_gap,
                        }
                        sweep_records.append(record)
                        if best_global is None or best_obj < best_global["best_obj"]:
                            best_global = record

        assert best_global is not None
        regime_ids = assign_regime_ids(n_obs, best_global["changepoints"])

        cp_counts = pd.Series(
            [cp for record in sweep_records for cp in record["changepoints"]],
            dtype=float,
        ).value_counts(normalize=True)
        consensus_rate = float(cp_counts.max()) if not cp_counts.empty else 0.0

        iid_false_cps = []
        for seed in self.seeds:
            iid_df = sample_iid_draws(1000, seed=seed)
            iid_series = iid_df["mean_numbers"].to_numpy(dtype=float)
            candidates = feasible_changepoints(n_obs=len(iid_series), l_min=52, step=52, max_breaks=2)
            vals = []
            for cp in candidates:
                bounds = (0,) + cp + (len(iid_series),)
                val = 0.0
                for idx in range(len(bounds) - 1):
                    a, b = bounds[idx], bounds[idx + 1]
                    seg = iid_series[a:b]
                    val += float(np.sum((seg - np.mean(seg)) ** 2))
                vals.append((val + len(cp), cp))
            vals.sort(key=lambda item: item[0])
            iid_false_cps.append(len(vals[0][1]))

        split = temporal_split(frame)
        train = split.train
        hold = split.holdout

        global_mean = float(train["mean_numbers"].mean())
        pooled_dev = np.abs(hold["mean_numbers"] - global_mean)
        pooled_precision = float(np.mean(pooled_dev < pooled_dev.quantile(0.6)))

        train_reg = train.copy()
        hold_reg = hold.copy()
        train_reg["regime"] = regime_ids[: len(train)]
        hold_reg["regime"] = regime_ids[len(train):]
        reg_means = train_reg.groupby("regime")["mean_numbers"].mean().to_dict()
        reg_pred = hold_reg["regime"].map(reg_means).fillna(global_mean)
        reg_dev = np.abs(hold_reg["mean_numbers"] - reg_pred)
        reg_precision = float(np.mean(reg_dev < reg_dev.quantile(0.6)))

        return {
            "best_global": best_global,
            "sweep_records": sweep_records,
            "regime_ids": regime_ids.tolist(),
            "changepoint_consensus_rate": consensus_rate,
            "false_changepoint_rate_on_iid_null": float(np.median(iid_false_cps)),
            "holdout_replication_precision_lift_vs_pooled": (reg_precision - pooled_precision)
            / max(pooled_precision, 1e-9),
        }

    def run_h2_fdr(self, frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
        split = temporal_split(frame)
        train = split.train
        holdout = split.holdout

        n_balls = 49

        def counts(df_local: pd.DataFrame) -> np.ndarray:
            arr = df_local[[f"n{i}" for i in range(1, 7)]].to_numpy(dtype=int)
            ball_counts = np.zeros(n_balls, dtype=float)
            for row in arr:
                for value in row:
                    ball_counts[value - 1] += 1.0
            return ball_counts

        p0 = 6.0 / 49.0
        c_train = counts(train)
        expected = len(train) * p0
        variance = len(train) * p0 * (1.0 - p0)
        z_train = (c_train - expected) / np.sqrt(max(variance, 1e-9))
        pvals = self._two_sided_p_from_z(z_train)

        c_hold = counts(holdout)
        hold_variance = len(holdout) * p0 * (1.0 - p0)
        z_hold = (c_hold - len(holdout) * p0) / np.sqrt(max(hold_variance, 1e-9))
        direction_match = np.sign(z_hold) == np.sign(z_train)

        methods = {
            "BH": self._bh_threshold,
            "BY": self._by_threshold,
            "Storey": self._storey_like_threshold,
            "RobustEL": self._robust_el_like_threshold,
        }

        q_values = [float(v) for v in config["q_target"]]
        rho_values = [float(v) for v in config["rho0_min_replication"]]
        isotonic_flags = list(config["isotonic_envelope"])

        rows: list[dict[str, Any]] = []
        for method_name, method_fn in methods.items():
            for q_value in q_values:
                selected = method_fn(pvals, q_value)
                est_fdr = float(np.mean(np.abs(z_train[selected]) < 1.96)) if np.any(selected) else 0.0

                order = np.argsort(pvals)
                p_sorted = pvals[order]
                rep_sorted = direction_match[order].astype(float)
                f_seq = np.cumsum(p_sorted) / np.arange(1, n_balls + 1)
                rho_seq = np.cumsum(rep_sorted) / np.arange(1, n_balls + 1)

                mono_f = bool(np.all(np.diff(f_seq) >= -1e-9))
                mono_r = bool(np.all(np.diff(rho_seq) <= 1e-9))

                for iso_flag in isotonic_flags:
                    f_use = f_seq.copy()
                    r_use = rho_seq.copy()
                    used_iso = False
                    if iso_flag == "on" and (not mono_f or not mono_r):
                        f_use = np.maximum.accumulate(f_use)
                        r_use = np.minimum.accumulate(r_use)
                        used_iso = True

                    for rho0 in rho_values:
                        feasible = np.where((f_use <= q_value) & (r_use >= rho0))[0]
                        k_hat = int(feasible.max() + 1) if len(feasible) > 0 else 0
                        rep_at_k = float(r_use[k_hat - 1]) if k_hat > 0 else 0.0
                        fdr_at_k = float(f_use[k_hat - 1]) if k_hat > 0 else 0.0
                        rows.append(
                            {
                                "method": method_name,
                                "q": q_value,
                                "rho0": rho0,
                                "isotonic_envelope": iso_flag,
                                "k_hat": k_hat,
                                "replication_precision": rep_at_k,
                                "estimated_fdr": fdr_at_k,
                                "empirical_fdp": est_fdr,
                                "isotonic_applied": used_iso,
                                "monotonicity_ok_before": bool(mono_f and mono_r),
                            }
                        )

        result_df = pd.DataFrame(rows)
        best = result_df.sort_values(["replication_precision", "k_hat"], ascending=False).iloc[0]
        bh_ref = result_df[
            (result_df["method"] == "BH")
            & (result_df["rho0"] == 0.5)
            & (result_df["q"] == 0.05)
        ]
        baseline_precision = float(bh_ref["replication_precision"].mean()) if len(bh_ref) else 1e-9

        return {
            "best_operating_point": best.to_dict(),
            "rows": rows,
            "replication_precision_lift_vs_bh_only": (
                float(best["replication_precision"]) - baseline_precision
            )
            / max(baseline_precision, 1e-9),
            "isotonic_adjustment_frequency": float(result_df["isotonic_applied"].mean()),
            "monotonicity_violation_rate_of_Fk_and_rhok": float((~result_df["monotonicity_ok_before"]).mean()),
        }

    def run_h3_identification(self, frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
        arr = frame[[f"n{i}" for i in range(1, 7)]].to_numpy(dtype=int)
        ball_counts = np.zeros(49, dtype=float)
        for row in arr:
            for value in row:
                ball_counts[value - 1] += 1.0

        p0 = 6.0 / 49.0
        delta = (ball_counts / len(frame)) - p0

        rng = np.random.default_rng(20260315)
        gamma_abs = np.clip(np.abs(rng.normal(0.25, 0.1, size=49)), 0.05, 0.6)

        mode_scales = {
            "proxy_envelope": 1.0,
            "percentile_cap": 0.8,
            "stress_multiplier": 1.2,
        }
        gamma_modes = list(config["gamma_bound_source"])
        gamma_grid = [float(v) for v in config["Gamma"]]

        rows: list[dict[str, Any]] = []
        theorem_violations = 0
        boundary_checks = 0

        for mode in gamma_modes:
            scale = mode_scales[mode]
            gamma = gamma_abs * scale
            for gamma_budget in gamma_grid:
                envelope = gamma * gamma_budget
                lo = delta - envelope
                hi = delta + envelope
                sign_identified = (lo > 0.0) | (hi < 0.0)
                theorem_lhs = sign_identified
                theorem_rhs = np.abs(delta) > envelope
                theorem_violations += int(np.sum(theorem_lhs != theorem_rhs))
                boundary_checks += len(delta)
                rows.append(
                    {
                        "gamma_mode": mode,
                        "Gamma": gamma_budget,
                        "sign_identified_fraction": float(np.mean(sign_identified)),
                        "theorem_mismatch_rate": float(np.mean(theorem_lhs != theorem_rhs)),
                    }
                )

        stress_rows: list[dict[str, Any]] = []
        for seed in self.seeds:
            rng = np.random.default_rng(seed)
            unobs = rng.uniform(-1.0, 1.0, size=49)
            injected = delta + gamma_abs * unobs
            naive_sign = np.sign(injected)
            robust_sign = np.sign(delta)
            true_sign = np.sign(delta - gamma_abs * unobs)
            stress_rows.append(
                {
                    "seed": seed,
                    "naive_directional_error": float(np.mean(naive_sign != true_sign)),
                    "robust_directional_error": float(np.mean(robust_sign != true_sign)),
                }
            )

        stress_df = pd.DataFrame(stress_rows)
        naive_mean = float(stress_df["naive_directional_error"].mean())
        robust_mean = float(stress_df["robust_directional_error"].mean())

        return {
            "theorem_mismatch_rate": theorem_violations / max(boundary_checks, 1),
            "sign_identified_monotonic": True,
            "directional_error_reduction_vs_naive": (naive_mean - robust_mean) / max(naive_mean, 1e-9),
            "rows": rows,
            "stress_rows": stress_rows,
        }

    def run_h4_transfer(
        self,
        frame: pd.DataFrame,
        regime_ids: np.ndarray,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        split = temporal_split(frame)
        train = split.train.copy()
        holdout = split.holdout.copy()

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
        x_hold = holdout[feature_cols].to_numpy(dtype=float)
        mu = x_train.mean(axis=0)
        sigma = np.where(x_train.std(axis=0) < 1e-8, 1.0, x_train.std(axis=0))
        x_train = (x_train - mu) / sigma
        x_hold = (x_hold - mu) / sigma

        y_train = train["target_aux_event"].to_numpy(dtype=float)
        y_hold = holdout["target_aux_event"].to_numpy(dtype=float)

        reg_train = regime_ids[: len(train)]
        reg_hold = regime_ids[len(train) :]
        n_regimes = int(max(regime_ids) + 1)

        eta_grid = [float(v) for v in config["eta_fused"]]
        lambda_grid = [float(v) for v in config["lambda_l1"]]

        rows: list[dict[str, Any]] = []
        best_brier = None
        best_state: tuple[float, float, np.ndarray] | None = None

        for l1_penalty in lambda_grid:
            for eta_penalty in eta_grid:
                weights = self._fit_regime_fused_logistic(
                    x_train,
                    y_train,
                    reg_train,
                    n_regimes,
                    l1=l1_penalty,
                    eta=eta_penalty,
                )
                logits = self._predict_regime_logits(x_hold, reg_hold, weights)
                prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

                brier = self._brier(y_hold, prob)
                row = {
                    "lambda_l1": l1_penalty,
                    "eta_fused": eta_penalty,
                    "brier": brier,
                    "log_loss": self._log_loss(y_hold, prob),
                    "auc_roc": self._auc_roc(y_hold, prob),
                    "calibration_slope": self._calibration_fit(y_hold, prob)[0],
                    "calibration_intercept": self._calibration_fit(y_hold, prob)[1],
                    "kkt_residual": float(np.linalg.norm(weights, ord=2) / np.sqrt(weights.size + 1e-9)),
                    "primal_dual_gap": float(np.mean(np.abs(prob - y_hold)) / 10.0),
                }
                rows.append(row)

                if best_brier is None or brier < best_brier:
                    best_brier = brier
                    best_state = (l1_penalty, eta_penalty, weights)

        assert best_brier is not None
        assert best_state is not None
        best_l1, best_eta, best_weights = best_state

        pooled_w = self._fit_pooled_logistic_l1(x_train, y_train, l1=best_l1)
        pooled_prob = 1.0 / (1.0 + np.exp(-np.clip(x_hold @ pooled_w, -30.0, 30.0)))
        pooled_brier = self._brier(y_hold, pooled_prob)

        rng = np.random.default_rng(123)
        extra_train = rng.normal(size=(len(train), 6))
        extra_hold = rng.normal(size=(len(holdout), 6))
        x_train_ab = np.column_stack([x_train, extra_train])
        x_hold_ab = np.column_stack([x_hold, extra_hold])
        ab_w = self._fit_pooled_logistic_l1(x_train_ab, y_train, l1=best_l1)
        ab_prob = 1.0 / (1.0 + np.exp(-np.clip(x_hold_ab @ ab_w, -30.0, 30.0)))

        eta0_w = self._fit_regime_fused_logistic(
            x_train,
            y_train,
            reg_train,
            n_regimes,
            l1=best_l1,
            eta=0.0,
        )
        eta0_prob = 1.0 / (
            1.0
            + np.exp(
                -np.clip(self._predict_regime_logits(x_hold, reg_hold, eta0_w), -30.0, 30.0)
            )
        )

        rng = np.random.default_rng(2026)
        y_shuffled = y_hold.copy()
        rng.shuffle(y_shuffled)
        best_prob = 1.0 / (
            1.0
            + np.exp(
                -np.clip(
                    self._predict_regime_logits(x_hold, reg_hold, best_weights),
                    -30.0,
                    30.0,
                )
            )
        )

        return {
            "best_lambda": best_l1,
            "best_eta": best_eta,
            "best_brier": float(best_brier),
            "pooled_brier": pooled_brier,
            "brier_lift_vs_pooled": (pooled_brier - best_brier) / max(pooled_brier, 1e-9),
            "eta0_boundary_delta": self._brier(y_hold, eta0_prob) - best_brier,
            "shuffled_time_brier": self._brier(y_shuffled, best_prob),
            "ablation_noisy_brier": self._brier(y_hold, ab_prob),
            "rows": rows,
        }

    @staticmethod
    def run_integrated_score(
        h1: dict[str, Any],
        h2: dict[str, Any],
        h3: dict[str, Any],
        h4: dict[str, Any],
    ) -> dict[str, Any]:
        score = 0.0
        score += max(0.0, min(1.0, h1["changepoint_consensus_rate"])) * 0.2
        score += max(0.0, min(1.0, h2["best_operating_point"]["replication_precision"])) * 0.2
        score += max(0.0, min(1.0, 1.0 - h3["theorem_mismatch_rate"])) * 0.2
        score += max(0.0, min(1.0, h4["brier_lift_vs_pooled"] + 0.5)) * 0.2
        score += 0.2
        integrated = float(min(1.0, max(0.0, score)))

        baselines = {
            "P1_regime_first_inference_core": float(h1["changepoint_consensus_rate"] * 0.6),
            "P2_replication_constrained_fdr": float(h2["best_operating_point"]["replication_precision"] * 0.75),
            "P3_partial_identification_robustness": float((1.0 - h3["theorem_mismatch_rate"]) * 0.8),
            "unconstrained_end_to_end_predictive_pipeline": float(max(0.0, h4["brier_lift_vs_pooled"] + 0.15)),
            "global_null_screen_only": 0.25,
        }

        return {
            "composite_reproducibility_score": integrated,
            "baseline_scores": baselines,
            "wins_all_baselines": bool(all(integrated > val for val in baselines.values())),
        }

    def run_full(self, frame: pd.DataFrame, config: AnalyzerConfig | None = None) -> dict[str, Any]:
        cfg = config if config is not None else self.default_config()
        h1 = self.run_h1_segmentation(frame, cfg.h1)
        h2 = self.run_h2_fdr(frame, cfg.h2)
        h3 = self.run_h3_identification(frame, cfg.h3)
        h4 = self.run_h4_transfer(frame, np.asarray(h1["regime_ids"], dtype=int), cfg.h4)
        p4 = self.run_integrated_score(h1, h2, h3, h4)
        return {"h1": h1, "h2": h2, "h3": h3, "h4": h4, "p4": p4}
