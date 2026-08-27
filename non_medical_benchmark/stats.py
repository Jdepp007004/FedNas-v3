"""Small, dependency-light statistical reporting helpers for the benchmark."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


# Two-sided 95% Student-t critical values for the small seed counts normally
# used in this repository.  Using t rather than 1.96 avoids overstating
# certainty when a run has only a handful of independent seeds.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
    19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def confidence_interval(
    values: Iterable[float],
    bounds: tuple[float, float] | None = None,
) -> dict[str, float | int | None]:
    """Return mean, sample standard deviation, and a two-sided 95% CI."""
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    n = int(array.size)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    mean = float(array.mean())
    if n == 1:
        return {"n": 1, "mean": mean, "std": 0.0, "ci95_low": None, "ci95_high": None}
    std = float(array.std(ddof=1))
    critical = _T95.get(n - 1, 1.96 if n > 120 else _T95[30])
    half_width = float(critical * std / math.sqrt(n))
    low = mean - half_width
    high = mean + half_width
    if bounds is not None:
        low = max(float(bounds[0]), low)
        high = min(float(bounds[1]), high)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_low": low,
        "ci95_high": high,
    }


def paired_interval(left: Iterable[float], right: Iterable[float]) -> dict[str, float | int | None]:
    """Report a paired right-minus-left interval for shared random seeds."""
    left_array = np.asarray(list(left), dtype=np.float64)
    right_array = np.asarray(list(right), dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise ValueError("paired arrays must have the same shape")
    return confidence_interval(right_array - left_array)
