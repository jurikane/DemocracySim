from __future__ import annotations

import numpy as np

from src.utils.social_welfare_functions import (
    APPROVAL_THRESHOLD_TAU,
    approval_voting,
    approval_voting_custom,
)


def test_approval_voting_canonical_fixed_threshold_oracle() -> None:
    """Canonical approval uses a fixed threshold tau on disagreement scores."""
    pref = np.array(
        [
            [0.71990938, 0.83556922, 0.28187783],
            [0.21521817, 0.63933138, 0.80505483],
            [0.96367087, 0.15052483, 0.48221239],
        ],
        dtype=np.float64,
    )
    # tau = 0.5 -> approvals:
    # col0:1, col1:1, col2:2 -> col2 must win.
    # Tie between col0 and col1 is resolved by lower total disagreement (col1).
    out = np.asarray(approval_voting(pref.copy(), rng=np.random.default_rng(1)), dtype=np.int64)
    assert list(out) == [2, 1, 0]


def test_approval_voting_custom_variant_is_available() -> None:
    """Legacy adaptive variant remains accessible for exploratory comparisons."""
    pref = np.array(
        [
            [0.8, 0.2, 0.1],
            [0.1, 0.9, 0.3],
            [0.3, 0.2, 0.7],
        ],
        dtype=np.float64,
    )
    out = np.asarray(approval_voting_custom(pref.copy(), rng=np.random.default_rng(7)), dtype=np.int64)
    assert out.shape == (pref.shape[1],)


def test_approval_voting_tie_randomness_under_full_symmetry() -> None:
    """With full symmetry, winners should vary across seeds."""
    pref = np.full((20, 4), APPROVAL_THRESHOLD_TAU, dtype=np.float64)
    winners = set()
    for seed in range(200):
        out = np.asarray(approval_voting(pref.copy(), rng=np.random.default_rng(seed)), dtype=np.int64)
        winners.add(int(out[0]))
    assert winners == {0, 1, 2, 3}
