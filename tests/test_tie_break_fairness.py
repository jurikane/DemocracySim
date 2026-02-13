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
def test_tie_break_fairness_no_structural_option_id_bias(rule) -> None:  # type: ignore[no-untyped-def]
    """Under a fully tied symmetric profile, winners should be approximately balanced."""
    m = 4
    pref_table = np.full((20, m), 0.5, dtype=np.float64)
    n = 400
    counts = np.zeros(m, dtype=np.int64)

    for seed in range(n):
        rng = np.random.default_rng(30000 + seed)
        winner = int(rule(pref_table.copy(), rng=rng)[0])
        counts[winner] += 1

    expected = n / m
    # Every option must win at least sometimes.
    assert np.all(counts > 0), f"Some options never won under full ties: {counts.tolist()}"
    # Loose but meaningful fairness bound.
    assert float(np.max(np.abs(counts - expected))) <= 25.0, (
        f"Tie-break appears structurally biased for {rule.__name__}: counts={counts.tolist()}"
    )
