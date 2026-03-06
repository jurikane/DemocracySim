from __future__ import annotations

import numpy as np
import pytest

from src.utils.social_welfare_functions import SCHULZE_MAX_OPTIONS, schulze_rule


pytestmark = pytest.mark.phase1


def test_schulze_returns_valid_full_permutation() -> None:
    pref = np.random.default_rng(1101).random((8, 6), dtype=np.float64)
    ordering = np.asarray(schulze_rule(pref, rng=np.random.default_rng(1201)), dtype=np.int64)
    assert ordering.shape == (6,)
    assert set(ordering.tolist()) == set(range(6))


def test_schulze_is_seed_deterministic_for_sequence() -> None:
    pref = np.random.default_rng(1102).random((7, 5), dtype=np.float64)
    rng_a = np.random.default_rng(4201)
    rng_b = np.random.default_rng(4201)
    seq_a = [tuple(np.asarray(schulze_rule(pref, rng=rng_a), dtype=np.int64).tolist()) for _ in range(10)]
    seq_b = [tuple(np.asarray(schulze_rule(pref, rng=rng_b), dtype=np.int64).tolist()) for _ in range(10)]
    assert seq_a == seq_b


def test_schulze_canonical_cycle_profile_has_expected_ranking() -> None:
    # 3x A>B>C, 2x B>C>A, 2x C>A>B => A>B>C under Schulze strongest paths.
    pref = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            [2.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [1.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    ordering = np.asarray(schulze_rule(pref, rng=np.random.default_rng(1203)), dtype=np.int64)
    assert ordering.tolist() == [0, 1, 2]


def test_schulze_exact_score_ties_are_neutral() -> None:
    # All voters tie candidates 0 and 1 exactly. Winner should not be label-biased.
    pref = np.zeros((11, 2), dtype=np.float64)
    trials = 1200
    counts = np.zeros(2, dtype=np.int64)
    for seed in range(trials):
        winner = int(schulze_rule(pref, rng=np.random.default_rng(88_000 + seed))[0])
        counts[winner] += 1
    expected = trials / 2.0
    assert float(np.max(np.abs(counts - expected))) <= 70.0, counts.tolist()


def test_schulze_fails_fast_above_max_options() -> None:
    pref = np.random.default_rng(1104).random((2, SCHULZE_MAX_OPTIONS + 1), dtype=np.float64)
    with pytest.raises(ValueError, match=f"max options={SCHULZE_MAX_OPTIONS}"):
        schulze_rule(pref, rng=np.random.default_rng(1204))


def test_schulze_handles_empty_shapes() -> None:
    pref_no_candidates = np.zeros((5, 0), dtype=np.float64)
    ordering0 = np.asarray(schulze_rule(pref_no_candidates, rng=np.random.default_rng(1205)), dtype=np.int64)
    assert ordering0.size == 0

    pref_no_voters = np.zeros((0, 4), dtype=np.float64)
    ordering1 = np.asarray(schulze_rule(pref_no_voters, rng=np.random.default_rng(1206)), dtype=np.int64)
    assert ordering1.tolist() == [0, 1, 2, 3]


def test_schulze_rejects_non_finite_input() -> None:
    pref = np.asarray([[0.0, 1.0], [np.nan, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="finite"):
        schulze_rule(pref, rng=np.random.default_rng(1207))
