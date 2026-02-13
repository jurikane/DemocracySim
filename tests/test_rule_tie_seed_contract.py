from __future__ import annotations

import numpy as np
import pytest

from src.utils.social_welfare_functions import (
    approval_voting,
    borda_rule,
    majority_rule,
    utilitarian_rule,
)


pytestmark = pytest.mark.phase1


@pytest.mark.parametrize(
    "rule",
    [majority_rule, approval_voting, utilitarian_rule, borda_rule],
)
def test_rule_tie_breaking_is_seed_deterministic(rule) -> None:
    pref = np.full((12, 5), 0.5, dtype=np.float64)
    o1 = rule(pref.copy(), rng=np.random.default_rng(77))
    o2 = rule(pref.copy(), rng=np.random.default_rng(77))
    np.testing.assert_array_equal(o1, o2)


@pytest.mark.parametrize(
    "rule",
    [majority_rule, approval_voting, utilitarian_rule, borda_rule],
)
def test_rule_tie_breaking_varies_across_seeds(rule) -> None:
    pref = np.full((12, 5), 0.5, dtype=np.float64)
    winners = []
    for seed in range(80):
        o = rule(pref.copy(), rng=np.random.default_rng(seed))
        winners.append(int(np.asarray(o, dtype=np.int64)[0]))
    assert len(set(winners)) > 1

