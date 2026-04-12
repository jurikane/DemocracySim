from __future__ import annotations

import numpy as np
import pytest

from src.models.participation_model import social_welfare_functions
from src.utils.social_welfare_functions import (
    approval_voting,
    borda_rule,
    plurality_rule,
    schulze_rule,
    random_rule,
    utilitarian_rule,
)


pytestmark = pytest.mark.phase1


def test_random_rule_returns_valid_full_permutation() -> None:
    pref = np.random.default_rng(11).random((9, 6), dtype=np.float64)
    ordering = np.asarray(random_rule(pref, rng=np.random.default_rng(123)), dtype=np.int64)
    assert ordering.shape == (6,)
    assert set(ordering.tolist()) == set(range(6))


def test_random_rule_is_seed_deterministic_for_sequence() -> None:
    pref = np.random.default_rng(12).random((7, 5), dtype=np.float64)

    rng_a = np.random.default_rng(4242)
    rng_b = np.random.default_rng(4242)

    seq_a = [tuple(np.asarray(random_rule(pref, rng=rng_a), dtype=np.int64).tolist()) for _ in range(12)]
    seq_b = [tuple(np.asarray(random_rule(pref, rng=rng_b), dtype=np.int64).tolist()) for _ in range(12)]

    assert seq_a == seq_b


def test_random_rule_winner_distribution_is_approximately_uniform() -> None:
    m = 4
    pref = np.zeros((15, m), dtype=np.float64)
    trials = 1200
    counts = np.zeros(m, dtype=np.int64)

    for seed in range(trials):
        winner = int(random_rule(pref, rng=np.random.default_rng(90_000 + seed))[0])
        counts[winner] += 1

    expected = trials / m
    # Loose fairness bound for multinomial variation (~4 sigma for this setting).
    assert float(np.max(np.abs(counts - expected))) <= 55.0, counts.tolist()


def test_random_rule_index_shift_keeps_canonical_order_and_inserts_schulze() -> None:
    names = [fn.__name__ for fn in social_welfare_functions]
    assert names[:5] == [
        "plurality_rule",
        "approval_voting",
        "utilitarian_rule",
        "borda_rule",
        "schulze_rule",
    ]
    assert names[5] == "random_rule"

    pref = np.asarray(
        [
            [0.1, 0.8, 0.9, 0.7],
            [0.2, 0.7, 0.8, 0.6],
            [0.3, 0.2, 0.9, 0.7],
            [0.4, 0.3, 0.8, 0.6],
            [0.5, 0.4, 0.7, 0.6],
        ],
        dtype=np.float64,
    )

    canonical = [plurality_rule, approval_voting, utilitarian_rule, borda_rule, schulze_rule]
    winners = [int(np.asarray(fn(pref.copy(), rng=np.random.default_rng(77)), dtype=np.int64)[0]) for fn in canonical]
    assert winners == [1, 0, 0, 0, 1]
