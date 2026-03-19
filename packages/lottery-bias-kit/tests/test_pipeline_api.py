from __future__ import annotations

import numpy as np

from lottery_bias_kit import LotteryBiasAnalyzer, sample_iid_draws


def test_full_pipeline_runs_and_returns_expected_keys() -> None:
    frame = sample_iid_draws(n_draws=320, seed=7)
    analyzer = LotteryBiasAnalyzer(seeds=[7])

    config = LotteryBiasAnalyzer.default_config()
    config.h1 = {"L_min_draws": [26], "beta": [1.0], "lambda_alarm": [0.0]}
    config.h2 = {
        "q_target": [0.05],
        "rho0_min_replication": [0.5],
        "isotonic_envelope": ["off"],
    }
    config.h3 = {
        "gamma_bound_source": ["proxy_envelope"],
        "Gamma": [0.0, 0.5],
    }
    config.h4 = {
        "eta_fused": [0.0],
        "lambda_l1": [0.001],
    }

    result = analyzer.run_full(frame, config=config)

    assert set(result.keys()) == {"h1", "h2", "h3", "h4", "p4"}
    assert "regime_ids" in result["h1"]
    assert "best_operating_point" in result["h2"]
    assert "theorem_mismatch_rate" in result["h3"]
    assert "best_brier" in result["h4"]
    assert "composite_reproducibility_score" in result["p4"]
    assert isinstance(result["h1"]["regime_ids"], list)
    assert np.isfinite(result["h4"]["best_brier"])
