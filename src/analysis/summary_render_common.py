from __future__ import annotations

import numpy as np
import pandas as pd

_Y_PAD_UNIT = 0.02
_Y_PAD_PERCENT = 1.5
_SMOOTH_WINDOW_STEPS = 9

def _set_unit_ylim_visible(ax, *, pad: float = _Y_PAD_UNIT) -> None:
    """Bounded [0,1] axis with a tiny pad so flat lines at 0/1 stay visible."""
    ax.set_ylim(-float(pad), 1.0 + float(pad))

def _set_percent_ylim_visible(ax, *, pad: float = _Y_PAD_PERCENT) -> None:
    """Bounded [0,100] axis with a tiny pad so flat lines at 0/100 stay visible."""
    ax.set_ylim(-float(pad), 100.0 + float(pad))

def _rolling_mean_nan(values: np.ndarray, *, window: int = _SMOOTH_WINDOW_STEPS) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    w = max(1, int(window))
    if w <= 1:
        return arr
    return (
        pd.Series(arr, dtype=float)
        .rolling(window=w, min_periods=1, center=True)
        .mean()
        .to_numpy(dtype=float)
    )

def _adjacent_abs_change_series(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    if arr.size == 1:
        return out
    cur = arr[1:]
    prev = arr[:-1]
    mask = np.isfinite(cur) & np.isfinite(prev)
    out[1:] = np.where(mask, np.abs(cur - prev), np.nan)
    return out
