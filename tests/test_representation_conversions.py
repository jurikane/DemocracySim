from __future__ import annotations

import numpy as np
import pytest

from src.utils.representations import (
    distribution_to_ordering,
    ordering_to_ranks,
    ranks_to_ordering,
    scores_to_ordering,
)


def test_ordering_rank_ordering_round_trip_property() -> None:
    rng = np.random.default_rng(8201)
    for n in (2, 3, 4, 6, 8):
        for _ in range(50):
            ordering = rng.permutation(n).astype(np.int64)
            back = ranks_to_ordering(ordering_to_ranks(ordering))
            assert np.array_equal(ordering, back)


def test_ranks_to_ordering_rejects_ties_fail_loud() -> None:
    with pytest.raises(ValueError, match="ties"):
        ranks_to_ordering(np.array([0, 1, 1, 2], dtype=np.int64))


def test_distribution_to_ordering_tie_behavior_contract() -> None:
    # With ties and no RNG: fail loudly (no biased fallback).
    d = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float64)
    with pytest.raises(ValueError, match="pass rng"):
        distribution_to_ordering(d, rng=None)

    # With RNG: ties are broken fairly (both tied top options should win often).
    wins = {0: 0, 1: 0}
    n = 300
    for seed in range(n):
        rng = np.random.default_rng(20000 + seed)
        top = int(distribution_to_ordering(d, rng=rng)[0])
        if top in wins:
            wins[top] += 1
    assert wins[0] > 0 and wins[1] > 0
    # Loose fairness bound to avoid flaky CI but still catch structural bias.
    assert abs(wins[0] - wins[1]) <= 80


def test_scores_to_ordering_requires_rng_when_scores_tie() -> None:
    s = np.array([0.2, 0.2, 0.7], dtype=np.float64)
    with pytest.raises(ValueError, match="pass rng"):
        scores_to_ordering(s, rng=None)


def test_distribution_to_ordering_requires_rng_only_for_ties() -> None:
    # No ties -> works without rng.
    distribution_to_ordering(np.array([0.7, 0.2, 0.1], dtype=np.float64), rng=None)

    # Ties and no RNG -> fail loudly.
    with pytest.raises(ValueError, match="pass rng"):
        distribution_to_ordering(np.array([0.5, 0.5, 0.0], dtype=np.float64), rng=None)
