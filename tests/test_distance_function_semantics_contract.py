from __future__ import annotations

import numpy as np
import pytest
from itertools import permutations
from math import comb

from src.utils.distance_functions import spearman_fr_order
from tests.factory import create_test_model


def _footrule_ordering(ordering_a: np.ndarray, ordering_b: np.ndarray) -> float:
    n = ordering_a.size
    rank_a = np.empty(n, dtype=np.int64)
    rank_b = np.empty(n, dtype=np.int64)
    rank_a[ordering_a] = np.arange(n)
    rank_b[ordering_b] = np.arange(n)
    dist = float(np.sum(np.abs(rank_a - rank_b)))
    if n % 2 == 0:
        max_dist = n**2 / 2
    else:
        max_dist = (n**2 - 1) / 2
    return dist / max_dist if max_dist > 0 else 0.0


@pytest.mark.parametrize("n", [3, 4])
def test_spearman_footrule_matches_ordering_semantics(n: int) -> None:
    perms = [np.asarray(p, dtype=np.int64) for p in permutations(range(n))]
    for a in perms:
        for b in perms:
            expected = _footrule_ordering(a, b)
            got = float(spearman_fr_order(a, b, None))
            assert got == pytest.approx(expected, abs=1e-12)


def test_color_search_pairs_are_full_combinations() -> None:
    model, _cfg = create_test_model(num_colors=5)
    pairs = list(model.color_search_pairs)
    assert len(pairs) == comb(model.num_colors, 2)
