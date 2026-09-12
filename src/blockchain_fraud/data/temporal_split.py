from __future__ import annotations

from math import floor
from typing import Iterable


def make_temporal_split(
    time_steps: Iterable[int],
    train_fraction: float = 0.60,
    val_fraction: float = 0.20,
) -> dict[str, list[int]]:
    steps = sorted({int(s) for s in time_steps})
    if len(steps) < 3:
        raise ValueError("Temporal split requires at least three unique time steps.")
    train_end = max(1, floor(train_fraction * len(steps)))
    val_end = max(train_end + 1, floor((train_fraction + val_fraction) * len(steps)))
    val_end = min(val_end, len(steps) - 1)
    return {
        "train_steps": steps[:train_end],
        "val_steps": steps[train_end:val_end],
        "test_steps": steps[val_end:],
    }

