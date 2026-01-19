from __future__ import annotations

import numpy as np

from src.utils.metrics import gini_index_0_100


def test_gini_index_0_100_edge_cases():
    assert gini_index_0_100([]) == 0
    assert gini_index_0_100([0]) == 0
    assert gini_index_0_100([0, 0, 0]) == 0
    assert gini_index_0_100([5]) == 0


def test_gini_index_0_100_basic_values():
    # perfect equality
    assert gini_index_0_100([10, 10, 10]) == 0

    # simple inequality: [0,0,10]
    # real gini = 2/3 => 66.6.. -> int(66)
    assert gini_index_0_100([0, 0, 10]) == 66

    # scale invariance
    assert gini_index_0_100([0, 0, 100]) == 66


def test_gini_index_0_100_numpy_input():
    arr = np.array([1, 2, 3, 4], dtype=float)
    # known gini for 1..4 is 0.25 => 25
    assert gini_index_0_100(arr) == 25

