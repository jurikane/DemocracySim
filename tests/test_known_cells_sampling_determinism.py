from __future__ import annotations

from typing import Any

from tests.factory import create_test_model


def _normalize_known_cell(x: Any) -> tuple[str, Any]:
    if isinstance(x, int):
        return ("color", int(x))
    return ("cell", tuple(getattr(x, "pos")))


def test_update_known_cells_is_deterministic_for_fixed_seed() -> None:
    """Known-cells sampling must be deterministic given the model seed.

    This protects voting-path determinism because estimate_real_distribution depends
    on agent.known_cells.
    """

    model1, _ = create_test_model(seed=123, num_colors=3, num_agents=5)
    model2, _ = create_test_model(seed=123, num_colors=3, num_agents=5)

    a1 = model1.voting_agents[0]
    a2 = model2.voting_agents[0]
    assert a1 is not None and a2 is not None

    area1 = model1.areas[0]
    area2 = model2.areas[0]
    assert area1 is not None and area2 is not None

    # Same seed => first sampling call should pick the same known-cell sample.
    a1.update_known_cells(area1)
    a2.update_known_cells(area2)

    k1 = [_normalize_known_cell(c) for c in a1.known_cells]
    k2 = [_normalize_known_cell(c) for c in a2.known_cells]

    assert k1 == k2
