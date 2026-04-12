from __future__ import annotations

import numpy as np
import pytest

from tests.factory import create_test_model
from src.utils.representations import (
    validate_ordering,
    validate_distribution,
)
from src.utils.social_welfare_functions import plurality_rule, approval_voting, utilitarian_rule, borda_rule


pytestmark = pytest.mark.phase1


def test_model_options_are_color_orderings() -> None:
    """Contract: model.options rows are ColorOrderings (permutations of colors)."""
    model, _cfg = create_test_model(seed=1, num_colors=4, num_agents=10, num_areas=2)
    opts = np.asarray(model.options)
    assert opts.ndim == 2
    assert opts.shape[1] == int(model.num_colors)
    for row in opts:
        validate_ordering(np.asarray(row, dtype=np.int64), int(model.num_colors))


def test_area_color_distribution_is_valid_distribution() -> None:
    """Contract: each area's color_distribution is a proper distribution (sum=1)."""
    model, _cfg = create_test_model(seed=2, num_colors=3, num_agents=20, num_areas=2, max_steps=2)
    for area in model.areas:
        validate_distribution(np.asarray(area.color_distribution, dtype=np.float64), int(model.num_colors))

    model.step()
    for area in model.areas:
        validate_distribution(np.asarray(area.color_distribution, dtype=np.float64), int(model.num_colors))


def test_area_voted_ordering_is_valid_when_set() -> None:
    """Contract: Area.voted_ordering (when not None) is a ColorOrdering."""
    model, _cfg = create_test_model(seed=3, num_colors=3, num_agents=30, num_areas=2, max_steps=2)
    model.step()
    for area in model.areas:
        if area.voted_ordering is None:
            continue
        validate_ordering(np.asarray(area.voted_ordering, dtype=np.int64), int(model.num_colors))


@pytest.mark.parametrize("rule", [plurality_rule, approval_voting, utilitarian_rule, borda_rule])
def test_social_welfare_functions_return_option_ordering(rule) -> None:  # type: ignore[no-untyped-def]
    """Contract: social welfare functions map a ScoreVector table -> OptionOrdering."""
    # Small option count: C=3 => options=6
    model, _cfg = create_test_model(seed=4, num_colors=3, num_agents=5, num_areas=1)
    m = int(model.options.shape[0])
    # Build a deterministic pref_table: 4 voters, 6 options.
    pref_table = np.linspace(0.0, 1.0, num=4 * m, dtype=np.float32).reshape((4, m))
    out = np.asarray(rule(pref_table, rng=model.voting_rng), dtype=np.int64)
    validate_ordering(out, m)


def test_estimate_real_distribution_handles_empty_known_cells() -> None:
    """Contract: estimate_real_distribution must not crash when known_cells is empty/None."""
    model, _cfg = create_test_model(seed=5, num_colors=4, num_agents=10, num_areas=1)
    agent = model.voting_agents[0]
    area = model.areas[0]

    # Force missing knowledge.
    agent.known_cells = [None] * int(model.known_cells)
    dist, conf = agent.estimate_real_distribution(area)

    dist = np.asarray(dist, dtype=np.float64)
    validate_distribution(dist, int(model.num_colors))
    assert conf == 0.0
