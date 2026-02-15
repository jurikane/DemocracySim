from __future__ import annotations

import pytest

from tests.factory import create_test_model


pytestmark = pytest.mark.phase1


def test_model_collections_are_dense_after_initialization() -> None:
    model, _ = create_test_model(num_agents=12, num_areas=4, num_colors=3)

    assert all(c is not None for c in model.color_cells)
    assert all(a is not None for a in model.voting_agents)
    assert all(a is not None for a in model.areas)

