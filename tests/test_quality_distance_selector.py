from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.quality_distance import resolve_quality_distance_series


def test_quality_distance_selector_uses_puzzle_distance_in_puzzle_mode() -> None:
    df = pd.DataFrame(
        {
            "dist_to_reality": [0.9, 0.8, 0.7],
            "puzzle_distance": [0.1, 0.2, 0.3],
        }
    )
    out = resolve_quality_distance_series(df, quality_target_mode="puzzle")
    np.testing.assert_allclose(out.to_numpy(dtype=float), np.asarray([0.1, 0.2, 0.3], dtype=float))


def test_quality_distance_selector_uses_reality_distance_in_reality_mode() -> None:
    df = pd.DataFrame(
        {
            "dist_to_reality": [0.9, 0.8, 0.7],
            "puzzle_distance": [0.1, 0.2, 0.3],
        }
    )
    out = resolve_quality_distance_series(df, quality_target_mode="reality")
    np.testing.assert_allclose(out.to_numpy(dtype=float), np.asarray([0.9, 0.8, 0.7], dtype=float))


def test_quality_distance_selector_fails_fast_for_missing_or_invalid_puzzle_distance() -> None:
    df_missing = pd.DataFrame({"dist_to_reality": [0.1, 0.2]})
    with pytest.raises(ValueError, match="puzzle_distance"):
        resolve_quality_distance_series(df_missing, quality_target_mode="puzzle")

    df_invalid = pd.DataFrame(
        {
            "dist_to_reality": [0.1, 0.2],
            "puzzle_distance": [0.3, float("nan")],
        }
    )
    with pytest.raises(ValueError, match="Non-finite puzzle_distance"):
        resolve_quality_distance_series(df_invalid, quality_target_mode="puzzle")

