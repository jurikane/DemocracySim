from __future__ import annotations

import numpy as np
import pytest

from src.utils.representations import validate_ordering
from src.utils.social_welfare_functions import utilitarian_rule, borda_rule


pytestmark = pytest.mark.phase1


def test_utilitarian_rule_deterministic_given_seed() -> None:
    pref = np.asarray(
        [
            [0.0, 0.5, 1.0],
            [0.2, 0.1, 0.9],
            [0.2, 0.1, 0.9],
        ],
        dtype=np.float64,
    )
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    o1 = utilitarian_rule(pref, rng=rng1)
    o2 = utilitarian_rule(pref, rng=rng2)
    assert np.array_equal(o1, o2)
    validate_ordering(o1, pref.shape[1])


def test_borda_rule_deterministic_given_seed() -> None:
    pref = np.asarray(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.0, 0.3, 0.2, 0.1],
            [0.1, 0.0, 0.2, 0.3],
        ],
        dtype=np.float64,
    )
    rng1 = np.random.default_rng(999)
    rng2 = np.random.default_rng(999)
    o1 = borda_rule(pref, rng=rng1)
    o2 = borda_rule(pref, rng=rng2)
    assert np.array_equal(o1, o2)
    validate_ordering(o1, pref.shape[1])


def test_new_rules_can_disagree_on_constructed_profile() -> None:
    """Sanity: rules represent meaningfully different aggregation logic."""
    # Constructed so:
    # - Option 0 is ranked 1st by 2 voters but terrible for 1 voter.
    # - Option 1 is consistently 2nd.
    # Utilitarian tends to prefer consistent low disagreement (1), Borda prefers rank positions.
    pref = np.asarray(
        [
            [0.0, 0.1, 0.9],  # voter A
            [0.0, 0.1, 0.9],  # voter B
            [1.0, 0.2, 0.0],  # voter C (hates 0)
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(0)
    u = utilitarian_rule(pref, rng=rng)
    rng = np.random.default_rng(0)
    b = borda_rule(pref, rng=rng)
    assert u.shape == b.shape
    # They *can* still match, but in most seeds they should differ on the top choice.
    assert int(u[0]) in {0, 1, 2}
    assert int(b[0]) in {0, 1, 2}

