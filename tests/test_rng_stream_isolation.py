from __future__ import annotations

import numpy as np
import pytest

from tests.factory import create_test_model
from src.utils.metrics import get_grid_colors


pytestmark = pytest.mark.phase1


def _snapshot_model_state(model) -> dict:
    # Grid snapshot
    grid = np.asarray(get_grid_colors(model))
    # Per-agent snapshot (sorted by id)
    agents = [a for a in (getattr(model, "voting_agents", []) or []) if a is not None]
    agents.sort(key=lambda a: int(a.unique_id))
    agent_rows = []
    for a in agents:
        agent_rows.append(
            (
                int(a.unique_id),
                float(a.assets),
                float(a.q_participation),
                float(a.altruism_factor),
                float(a.dissatisfaction_value),
            )
        )
    # Per-area snapshot
    areas = [a for a in (getattr(model, "areas", []) or []) if a is not None]
    areas.sort(key=lambda a: int(a.unique_id))
    area_rows = []
    for area in areas:
        vo = None
        if area.voted_ordering is not None:
            vo = tuple(int(x) for x in np.asarray(area.voted_ordering).tolist())
        area_rows.append((int(area.unique_id), float(area.voter_turnout), float(area.dist_to_reality), vo))

    return {
        "grid": grid,
        "global_color_dst": np.asarray(model.global_color_dst, dtype=np.float64),
        "agents": agent_rows,
        "areas": area_rows,
    }


def test_consuming_viz_and_debug_rng_does_not_change_simulation() -> None:
    """Contract: viz/debug RNG streams must not perturb main simulation RNG."""
    overrides = dict(
        seed=202,
        num_colors=3,
        num_agents=40,
        num_areas=2,
        height=10,
        width=10,
        av_area_height=10,
        av_area_width=5,
        area_size_variance=0.0,
        max_steps=4,
    )
    m1, _ = create_test_model(**overrides)
    m2, _ = create_test_model(**overrides)

    # Consume viz/debug streams heavily on m2.
    _ = m2.rng_viz.random(10_000)
    _ = m2.rng_debug.random(10_000)
    for _i in range(10_000):
        m2.random_viz.random()
        m2.random_debug.random()

    # Step both models the same number of times.
    for _ in range(3):
        m1.step()
        m2.step()

    s1 = _snapshot_model_state(m1)
    s2 = _snapshot_model_state(m2)

    np.testing.assert_array_equal(s1["grid"], s2["grid"])
    np.testing.assert_allclose(s1["global_color_dst"], s2["global_color_dst"], rtol=0, atol=0)
    assert s1["agents"] == s2["agents"]
    assert s1["areas"] == s2["areas"]


def test_debug_panel_enabled_does_not_change_simulation() -> None:
    """Contract: enabling the live debug panel must not change dynamics."""
    overrides = dict(
        seed=303,
        num_colors=4,
        num_agents=60,
        num_areas=2,
        height=10,
        width=10,
        av_area_height=10,
        av_area_width=5,
        area_size_variance=0.0,
        max_steps=4,
    )
    base, _ = create_test_model(**overrides)
    dbg, _ = create_test_model(**overrides)

    # Mimic viz.debug_viz enabling.
    setattr(dbg, "_debug_agent_panel_enabled", True)
    setattr(dbg, "_debug_agent_panel_max_steps", 5)

    for _ in range(3):
        base.step()
        dbg.step()

    s1 = _snapshot_model_state(base)
    s2 = _snapshot_model_state(dbg)
    np.testing.assert_array_equal(s1["grid"], s2["grid"])
    np.testing.assert_allclose(s1["global_color_dst"], s2["global_color_dst"], rtol=0, atol=0)
    assert s1["agents"] == s2["agents"]
    assert s1["areas"] == s2["areas"]

