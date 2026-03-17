from __future__ import annotations

from lottery_bias_kit import LotteryBiasAnalyzer, sample_iid_draws


def main() -> None:
    frame = sample_iid_draws(n_draws=320, seed=11)
    analyzer = LotteryBiasAnalyzer(seeds=[11])

    config = LotteryBiasAnalyzer.default_config()
    result = analyzer.run_full(frame, config=config)

    print("Integrated score:", round(result["p4"]["composite_reproducibility_score"], 4))
    print("H4 lift vs pooled:", round(result["h4"]["brier_lift_vs_pooled"], 4))


if __name__ == "__main__":
    main()
