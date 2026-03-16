from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitData:
    train: pd.DataFrame
    holdout: pd.DataFrame


def load_lottery_draws(path: str | Path) -> pd.DataFrame:
    """Load historical lottery draws and enforce canonical numeric/date fields."""
    frame = pd.read_csv(Path(path))
    return prepare_lottery_dataframe(frame)


def prepare_lottery_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort draws by date and derive reusable number-theoretic descriptors."""
    out = frame.copy()
    num_cols = [f"n{i}" for i in range(1, 7)]

    for col in num_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=num_cols + ["draw_date"]).copy()
    out["draw_date"] = pd.to_datetime(out["draw_date"])
    out = out.sort_values("draw_date").reset_index(drop=True)
    out["draw_index"] = np.arange(len(out), dtype=int)

    nums = out[num_cols].to_numpy(dtype=float)
    sorted_nums = np.sort(nums, axis=1)

    out["sum_numbers"] = sorted_nums.sum(axis=1)
    out["mean_numbers"] = sorted_nums.mean(axis=1)
    out["odd_count"] = (sorted_nums % 2).sum(axis=1)
    out["low_count"] = (sorted_nums <= 24).sum(axis=1)
    out["span"] = sorted_nums[:, -1] - sorted_nums[:, 0]
    out["consecutive_pairs"] = (np.diff(sorted_nums, axis=1) == 1).sum(axis=1)

    sets = [set(row.astype(int).tolist()) for row in sorted_nums]
    overlaps = [0]
    for idx in range(1, len(sets)):
        overlaps.append(len(sets[idx].intersection(sets[idx - 1])))
    out["overlap_prev"] = overlaps

    mod3 = sorted_nums.astype(int) % 3
    out["mod3_count_0"] = (mod3 == 0).sum(axis=1)
    out["mod3_count_1"] = (mod3 == 1).sum(axis=1)
    out["mod3_count_2"] = (mod3 == 2).sum(axis=1)
    out["target_aux_event"] = (out["overlap_prev"] >= 2).astype(int)
    return out


def temporal_split(frame: pd.DataFrame, holdout_frac: float = 0.3) -> SplitData:
    """Time-ordered train/holdout split with robust bounds for small datasets."""
    n_rows = len(frame)
    split_idx = int(round(n_rows * (1.0 - holdout_frac)))
    split_idx = max(200, min(split_idx, n_rows - 50))
    return SplitData(
        train=frame.iloc[:split_idx].copy(),
        holdout=frame.iloc[split_idx:].copy(),
    )


def sample_iid_draws(
    n_draws: int,
    n_balls: int = 49,
    pick: int = 6,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate synthetic IID lottery draws and derived descriptors."""
    rng = np.random.default_rng(seed)
    rows: list[list[int]] = []
    for _ in range(n_draws):
        draw = np.sort(rng.choice(np.arange(1, n_balls + 1), size=pick, replace=False))
        rows.append(draw.tolist())

    out = pd.DataFrame(rows, columns=[f"n{i}" for i in range(1, pick + 1)])
    out["draw_date"] = pd.date_range("2000-01-01", periods=n_draws, freq="7D")
    out["year"] = out["draw_date"].dt.year
    return prepare_lottery_dataframe(out)


def feasible_changepoints(
    n_obs: int,
    l_min: int,
    step: int = 26,
    max_breaks: int = 2,
) -> list[tuple[int, ...]]:
    """Enumerate changepoints respecting minimum segment length constraints."""
    grid = list(range(l_min, n_obs - l_min + 1, step))
    feasible: list[tuple[int, ...]] = [tuple()]
    for n_breaks in range(1, max_breaks + 1):
        for cp in combinations(grid, n_breaks):
            bounds = (0,) + cp + (n_obs,)
            if all((bounds[i + 1] - bounds[i]) >= l_min for i in range(len(bounds) - 1)):
                feasible.append(cp)
    return feasible


def assign_regime_ids(n_obs: int, changepoints: Iterable[int]) -> np.ndarray:
    """Assign integer regime id for each time index based on sorted changepoints."""
    cps = sorted(int(c) for c in changepoints)
    regime = np.zeros(n_obs, dtype=int)
    current = 0
    cp_idx = 0
    for idx in range(n_obs):
        while cp_idx < len(cps) and idx >= cps[cp_idx]:
            current += 1
            cp_idx += 1
        regime[idx] = current
    return regime
