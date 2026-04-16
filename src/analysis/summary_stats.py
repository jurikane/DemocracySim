from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def time_mean(values: np.ndarray | list[float]) -> float:
    arr = np.asarray(values, dtype=float).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def step_volatility_l1_normalized(
    values: np.ndarray | list[float],
    *,
    value_range: float,
) -> float:
    if not np.isfinite(value_range) or value_range <= 0.0:
        raise ValueError("value_range must be finite and > 0.")

    arr = np.asarray(values, dtype=float).ravel()
    if arr.size < 2:
        return float("nan")

    left = arr[:-1]
    right = arr[1:]
    valid = np.isfinite(left) & np.isfinite(right)
    if not valid.any():
        return float("nan")

    return float(np.abs(right[valid] - left[valid]).mean() / value_range)

def _build_summary_stats(
    *,
    global_series: pd.DataFrame,
    area_series: pd.DataFrame,
    meta: dict[str, Any],
    static: dict[str, Any],
) -> dict[str, Any]:
    def _safe_final(series: pd.Series) -> float:
        arr = series.to_numpy(dtype=float)
        if arr.size == 0:
            return float("nan")
        return float(arr[-1])

    summary = {
        "run": {
            "run_seed": int(meta["run"]["run_seed"]),
            "rule_idx": int(meta["run"]["rule_idx"]),
            "rule_name": meta["run"].get("rule_name"),
            "distance_name": meta["run"].get("distance_name"),
        },
        "shape": {
            "num_steps": int(global_series["step"].nunique()),
            "num_areas": int(area_series["area_id"].nunique()),
            "num_agents": int(static.get("num_agents", 0)),
            "num_colors": int(static.get("num_colors", 0)),
        },
        "global_summary": {
            "turnout_mean": time_mean(global_series["turnout"].to_numpy(dtype=float)),
            "turnout_final": _safe_final(global_series["turnout"]),
            "turnout_volatility": step_volatility_l1_normalized(
                global_series["turnout"].to_numpy(dtype=float),
                value_range=100.0,
            ),
            "gini_assets_mean": time_mean(global_series["gini_assets"].to_numpy(dtype=float)),
            "gini_assets_final": _safe_final(global_series["gini_assets"]),
            "gini_assets_volatility": step_volatility_l1_normalized(
                global_series["gini_assets"].to_numpy(dtype=float),
                value_range=100.0,
            ),
            "gini_dissatisfaction_mean": time_mean(global_series["gini_dissatisfaction"].to_numpy(dtype=float)),
            "gini_dissatisfaction_final": _safe_final(global_series["gini_dissatisfaction"]),
            "gini_dissatisfaction_volatility": step_volatility_l1_normalized(
                global_series["gini_dissatisfaction"].to_numpy(dtype=float),
                value_range=100.0,
            ),
            "mean_dissatisfaction_mean": time_mean(global_series["mean_dissatisfaction"].to_numpy(dtype=float)),
            "mean_dissatisfaction_final": _safe_final(global_series["mean_dissatisfaction"]),
            "quality_distance_mean": time_mean(global_series["quality_distance"].to_numpy(dtype=float)),
            "quality_distance_final": _safe_final(global_series["quality_distance"]),
            "quality_distance_volatility": step_volatility_l1_normalized(
                global_series["quality_distance"].to_numpy(dtype=float),
                value_range=1.0,
            ),
            "diversity_entropy_mean": time_mean(global_series["diversity_first_choice_entropy"].to_numpy(dtype=float)),
            "diversity_entropy_final": _safe_final(global_series["diversity_first_choice_entropy"]),
        },
    }
    return summary
