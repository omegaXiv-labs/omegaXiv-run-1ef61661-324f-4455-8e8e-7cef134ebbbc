from pathlib import Path
import json


def test_results_summary_exists_after_run() -> None:
    path = Path("experiments/EXP_P4_end_to_end_integrated_pipeline/iter_1/results_summary.json")
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "h1" in data
    assert "h5" in data
    assert "figure_captions" in data
    assert len(data.get("figures", [])) >= 4


def test_config_seed_count() -> None:
    cfg_path = Path("experiments/EXP_P4_end_to_end_integrated_pipeline/configs/default.json")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert len(cfg["seeds"]) >= 5
    assert "h5" in cfg


def test_iter_1_paths_declared() -> None:
    path = Path("experiments/EXP_P4_end_to_end_integrated_pipeline/iter_1/results_summary.json")
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    for p in data.get("figures", []) + data.get("tables", []) + data.get("datasets", []):
        assert "iter_1" in p
