from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from src.models.participation_model import distance_functions


pytestmark = pytest.mark.phase1


@pytest.mark.parametrize("distance_idx", list(range(len(distance_functions))))
@pytest.mark.parametrize("n", [0, 1, 3, 4])
def test_distance_functions_basic_contract(distance_idx: int, n: int) -> None:
    """Contract for ordering distances used in ballots + rewards.

    For ColorOrderings (permutations):
    - returns finite float in [0,1]
    - symmetric
    - d(x,x)=0
    - for n<=1 returns 0 (no division by zero)
    """
    f = distance_functions[distance_idx]
    if n == 0:
        a = np.asarray([], dtype=np.int64)
        b = np.asarray([], dtype=np.int64)
        pairs = []
    else:
        a = np.arange(n, dtype=np.int64)
        b = np.arange(n, dtype=np.int64)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    d0 = float(f(a, a, pairs))
    assert np.isfinite(d0)
    assert 0.0 <= d0 <= 1.0
    assert d0 == pytest.approx(0.0)

    d1 = float(f(a, b, pairs))
    d2 = float(f(b, a, pairs))
    assert np.isfinite(d1) and np.isfinite(d2)
    assert 0.0 <= d1 <= 1.0
    assert 0.0 <= d2 <= 1.0
    assert d1 == pytest.approx(d2)


@pytest.mark.parametrize("distance_idx", list(range(len(distance_functions))))
def test_distance_functions_full_range_on_small_perms(distance_idx: int) -> None:
    """For small n, scan all permutation pairs and ensure values stay in [0,1]."""
    f = distance_functions[distance_idx]
    n = 4
    perms = [np.asarray(p, dtype=np.int64) for p in permutations(range(n))]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    mx = 0.0
    for a in perms:
        for b in perms:
            d = float(f(a, b, pairs))
            assert np.isfinite(d)
            assert 0.0 <= d <= 1.0
            mx = max(mx, d)
    # Normalized distances should be capable of reaching (close to) 1 on reverse orderings.
    assert mx >= 0.99

