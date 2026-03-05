from __future__ import annotations

import numpy as np


def _finite_1d(x: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def time_mean(x: np.ndarray | list[float]) -> float:
    """Mean over finite time points; NaN if no finite values."""
    vals = _finite_1d(x)
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def time_mean_last_frac(x: np.ndarray | list[float], *, frac: float = 0.2) -> float:
    """Mean over the last `frac` of points (finite only); NaN if none."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    n = int(arr.size)
    if n <= 0:
        return float("nan")
    f = float(frac)
    if not np.isfinite(f) or f <= 0.0:
        raise ValueError("frac must be finite and > 0.")
    k = max(1, int(np.ceil(f * n)))
    tail = arr[-k:]
    vals = tail[np.isfinite(tail)]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def early_late_delta(
    x: np.ndarray | list[float],
    *,
    early_frac: float = 0.2,
    late_frac: float = 0.2,
) -> float:
    """late_mean - early_mean using finite points in the selected windows."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    n = int(arr.size)
    if n <= 0:
        return float("nan")
    ef = float(early_frac)
    lf = float(late_frac)
    if (not np.isfinite(ef)) or (not np.isfinite(lf)) or ef <= 0.0 or lf <= 0.0:
        raise ValueError("early_frac and late_frac must be finite and > 0.")
    k_e = max(1, int(np.ceil(ef * n)))
    k_l = max(1, int(np.ceil(lf * n)))
    early = arr[:k_e]
    late = arr[-k_l:]
    early_f = early[np.isfinite(early)]
    late_f = late[np.isfinite(late)]
    if early_f.size == 0 or late_f.size == 0:
        return float("nan")
    return float(np.mean(late_f) - np.mean(early_f))


def step_volatility_l1(x: np.ndarray | list[float]) -> float:
    """Mean absolute adjacent-step change over finite pairs; NaN if <1 valid pair."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size < 2:
        return float("nan")
    left = arr[:-1]
    right = arr[1:]
    mask = np.isfinite(left) & np.isfinite(right)
    if not np.any(mask):
        return float("nan")
    diffs = np.abs(right[mask] - left[mask])
    if diffs.size == 0:
        return float("nan")
    return float(np.mean(diffs))


def step_volatility_l1_normalized(
    x: np.ndarray | list[float],
    *,
    value_range: float,
) -> float:
    """Normalized step volatility: step_volatility_l1(x) / value_range.

    No clamping is applied by design to avoid masking scale mistakes.
    """
    r = float(value_range)
    if not np.isfinite(r) or r <= 0.0:
        raise ValueError("value_range must be finite and > 0.")
    v = step_volatility_l1(x)
    if not np.isfinite(v):
        return float("nan")
    return float(v / r)

