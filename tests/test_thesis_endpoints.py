from __future__ import annotations

import math

import numpy as np
import pytest

from src.analysis.thesis_endpoints import (
    early_late_delta,
    step_volatility_l1,
    step_volatility_l1_normalized,
    time_mean,
    time_mean_last_frac,
)


def test_step_volatility_l1_constant_series_is_zero() -> None:
    x = np.asarray([3.0, 3.0, 3.0, 3.0], dtype=float)
    assert step_volatility_l1(x) == pytest.approx(0.0, abs=0.0)


def test_step_volatility_l1_alternating_series() -> None:
    x = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=float)
    assert step_volatility_l1(x) == pytest.approx(1.0, abs=1e-12)


def test_step_volatility_l1_monotone_ramp() -> None:
    x = np.asarray([0.0, 0.5, 1.0], dtype=float)
    assert step_volatility_l1(x) == pytest.approx(0.5, abs=1e-12)


def test_step_volatility_l1_skips_nonfinite_pairs_single_nan_middle() -> None:
    x = np.asarray([0.0, np.nan, 0.5, 1.0], dtype=float)
    # Valid adjacent pairs: only (0.5, 1.0) => |diff| = 0.5
    assert step_volatility_l1(x) == pytest.approx(0.5, abs=1e-12)


def test_step_volatility_l1_returns_nan_if_no_valid_pair() -> None:
    assert math.isnan(step_volatility_l1(np.asarray([1.0], dtype=float)))
    assert math.isnan(step_volatility_l1(np.asarray([np.nan, 1.0], dtype=float)))


def test_step_volatility_l1_normalized_no_clamp() -> None:
    x = np.asarray([0.0, 150.0], dtype=float)
    # No clamp by design; normalization reflects raw scale mismatch.
    assert step_volatility_l1_normalized(x, value_range=100.0) == pytest.approx(1.5, abs=1e-12)


def test_time_means_and_early_late_delta_nan_policy() -> None:
    x = np.asarray([np.nan, 1.0, 2.0, np.nan, 3.0], dtype=float)
    assert time_mean(x) == pytest.approx(2.0, abs=1e-12)
    assert time_mean_last_frac(x, frac=0.4) == pytest.approx(3.0, abs=1e-12)
    # early window => [nan,1.0], late window => [nan,3.0]
    assert early_late_delta(x, early_frac=0.4, late_frac=0.4) == pytest.approx(2.0, abs=1e-12)

