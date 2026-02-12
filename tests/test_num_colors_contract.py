from __future__ import annotations

import numpy as np
import pytest

from tests.factory import create_test_model
from src.utils.representations import validate_ordering


pytestmark = pytest.mark.phase1


def test_num_colors_oracle_options_are_all_permutations_for_small_n() -> None:
    n = 4
    model, _ = create_test_model(
        seed=500,
        num_colors=n,
        num_personality_groups=n,
        num_agents=10,
        num_areas=1,
        height=8,
        width=8,
        av_area_height=8,
        av_area_width=8,
        area_size_variance=0.0,
        mu=0.0,
        known_cells=0,
        max_steps=1,
    )
    opts = np.asarray(model.options)
    assert opts.shape == (24, n)
    # Each row is a ColorOrdering permutation of [0..n-1]
    for row in opts:
        validate_ordering(np.asarray(row, dtype=np.int64), n)


def test_num_colors_metamorphic_factorial_growth_ratio() -> None:
    """Metamorphic: number of options grows factorially; (n+1)! / n! == (n+1)."""
    m3, _ = create_test_model(seed=501, num_colors=3, num_personality_groups=3, num_agents=5, num_areas=1, known_cells=0, mu=0.0)
    m4, _ = create_test_model(seed=501, num_colors=4, num_personality_groups=4, num_agents=5, num_areas=1, known_cells=0, mu=0.0)
    c3 = int(np.asarray(m3.options).shape[0])
    c4 = int(np.asarray(m4.options).shape[0])
    assert c3 == 6
    assert c4 == 24
    assert (c4 // c3) == 4


def test_num_colors_validation_minimum_two() -> None:
    with pytest.raises(ValueError, match=r"num_colors must be >= 2"):
        create_test_model(seed=502, num_colors=1)


def test_num_colors_validation_explosion_cap() -> None:
    # 9! = 362880 > 50000 cap => must raise early.
    with pytest.raises(ValueError, match=r"num_colors=9 implies 362880 options"):
        create_test_model(seed=503, num_colors=9)

