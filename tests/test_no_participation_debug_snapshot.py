from __future__ import annotations

import numpy as np

from tests.factory import create_test_model


class _NeverParticipate:
    def decide_participation(self, agent, area) -> bool:  # type: ignore[no-untyped-def]
        return False


def test_no_participation_debug_snapshot_records_outcome_fields() -> None:
    model, _ = create_test_model(
        seed=123,
        num_colors=3,
        num_agents=10,
        num_areas=1,
        height=8,
        width=8,
        av_area_height=8,
        av_area_width=8,
        area_size_variance=0.0,
        max_steps=2,
    )
    # Enable debug capture.
    setattr(model, "_debug_agent_panel_enabled", True)
    setattr(model, "_debug_agent_panel_max_steps", 5)

    for a in model.voting_agents:
        if a is None:
            continue
        a.participation_strategy = _NeverParticipate()

    area = model.areas[0]
    # Avoid ties so `real_color_ord` can be matched deterministically.
    area._color_distribution = np.asarray([0.1, 0.3, 0.6], dtype=np.float64)

    model.step()

    hist = area.debug_history
    assert hist, "Expected at least one debug snapshot"
    rec = hist[-1]

    assert rec["num_participants"] == 0
    assert rec["election_held"] is False
    # Carried-over outcome should still be resolvable to an option id.
    assert rec["winning_option_id"] is not None
    # Under no participation, this field is expected to remain None.
    assert rec["winning_option"] is None

