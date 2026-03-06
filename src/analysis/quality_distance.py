from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def normalize_quality_target_mode(value: Any) -> str:
    """Normalize configured quality target mode to canonical string."""
    if isinstance(value, (int, np.integer, float, np.floating)):
        return "puzzle" if int(value) == 1 else "reality"
    return str(value).strip().lower()


def resolve_quality_distance_series(
    table: pd.DataFrame,
    *,
    quality_target_mode: str,
    run_label: str = "",
) -> pd.Series:
    """Return canonical quality_distance for each table row.

    Contract:
    - puzzle mode: requires finite `puzzle_distance` for all rows.
    - reality mode: requires finite `dist_to_reality` for all rows.
    - no silent fallback between metrics.
    """
    mode = normalize_quality_target_mode(quality_target_mode)
    suffix = f" ({run_label})" if run_label else ""
    if mode == "puzzle":
        if "puzzle_distance" not in table.columns:
            raise ValueError(f"Missing required column 'puzzle_distance' for puzzle mode{suffix}.")
        s = pd.to_numeric(table["puzzle_distance"], errors="coerce")
        bad = ~np.isfinite(s.to_numpy(dtype=float))
        if bool(np.any(bad)):
            raise ValueError(
                f"Non-finite puzzle_distance encountered in puzzle mode{suffix}; "
                "this is invalid for quality-gate analysis."
            )
        return s.astype(float)
    if mode == "reality":
        if "dist_to_reality" not in table.columns:
            raise ValueError(f"Missing required column 'dist_to_reality' for reality mode{suffix}.")
        s = pd.to_numeric(table["dist_to_reality"], errors="coerce")
        bad = ~np.isfinite(s.to_numpy(dtype=float))
        if bool(np.any(bad)):
            raise ValueError(
                f"Non-finite dist_to_reality encountered in reality mode{suffix}; "
                "this is invalid for quality-gate analysis."
            )
        return s.astype(float)
    raise ValueError(f"Unsupported quality_target_mode: {quality_target_mode!r}")


def quality_distance_source(quality_target_mode: str) -> str:
    mode = normalize_quality_target_mode(quality_target_mode)
    if mode == "puzzle":
        return "puzzle_distance"
    if mode == "reality":
        return "dist_to_reality"
    raise ValueError(f"Unsupported quality_target_mode: {quality_target_mode!r}")

