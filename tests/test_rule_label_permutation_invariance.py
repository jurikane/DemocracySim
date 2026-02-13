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
def test_rule_is_neutral_under_option_relabeling(rule) -> None:
    """
    Metamorphic contract:
    relabel options (permute columns), run rule, map winner back -> same winner.

    We only assert winner-level neutrality on profiles with a stable winner
    (i.e., where changing seed does not alter the top option), so tie-breaking
    randomness in lower ranks does not make this test flaky.
    """
    rng = np.random.default_rng(2026)
    checked = 0
    for _ in range(80):
        n = int(rng.integers(5, 15))
        m = int(rng.integers(3, 7))
        pref = rng.random((n, m), dtype=np.float64)
        pref += np.arange(m, dtype=np.float64) * 1e-12
        perm = rng.permutation(m)
        pref_perm = pref[:, perm]

        # Keep only profiles where winner is seed-stable (likely non-tied top).
        w1 = int(np.asarray(rule(pref.copy(), rng=np.random.default_rng(1)), dtype=np.int64)[0])
        w2 = int(np.asarray(rule(pref.copy(), rng=np.random.default_rng(2)), dtype=np.int64)[0])
        if w1 != w2:
            continue

        w_orig = int(np.asarray(rule(pref.copy(), rng=np.random.default_rng(1234)), dtype=np.int64)[0])
        w_perm = int(np.asarray(rule(pref_perm.copy(), rng=np.random.default_rng(1234)), dtype=np.int64)[0])
        mapped_winner = int(perm[w_perm])
        assert mapped_winner == w_orig
        checked += 1

    assert checked >= 8
