from __future__ import annotations

import numpy as np
import pytest

from tests.factory import create_test_model


pytestmark = pytest.mark.phase1


def test_area_tally_votes_raises_on_wrong_scorevector_shape(monkeypatch) -> None:
    # Ensure the single area covers the whole grid so it contains agents.
    model, _cfg = create_test_model(
        seed=11,
        num_colors=3,
        num_agents=1,
        num_areas=1,
        av_area_width=10,
        av_area_height=10,
        width=10,
        height=10,
        area_size_variance=0.0,
    )
    area = model.areas[0]
    assert area.agents, "Expected the area to contain at least one agent for this contract test"
    agent = area.agents[0]

    monkeypatch.setattr(agent, "ask_for_participation", lambda *, area: True)
    monkeypatch.setattr(agent, "vote", lambda *, area: np.asarray([0.1, 0.2], dtype=np.float32))

    with pytest.raises(ValueError, match="score vector"):
        area._tally_votes()


def test_area_tally_votes_raises_on_non_finite_scores(monkeypatch) -> None:
    model, _cfg = create_test_model(
        seed=12,
        num_colors=3,
        num_agents=1,
        num_areas=1,
        av_area_width=10,
        av_area_height=10,
        width=10,
        height=10,
        area_size_variance=0.0,
    )
    area = model.areas[0]
    assert area.agents, "Expected the area to contain at least one agent for this contract test"
    agent = area.agents[0]
    m = int(model.options.shape[0])

    bad = np.zeros(m, dtype=np.float32)
    bad[0] = np.nan

    monkeypatch.setattr(agent, "ask_for_participation", lambda *, area: True)
    monkeypatch.setattr(agent, "vote", lambda *, area: bad)

    with pytest.raises(ValueError, match="finite"):
        area._tally_votes()


def test_area_tally_votes_raises_on_out_of_range_scores(monkeypatch) -> None:
    model, _cfg = create_test_model(
        seed=13,
        num_colors=3,
        num_agents=1,
        num_areas=1,
        av_area_width=10,
        av_area_height=10,
        width=10,
        height=10,
        area_size_variance=0.0,
    )
    area = model.areas[0]
    assert area.agents
    agent = area.agents[0]
    m = int(model.options.shape[0])

    bad = np.zeros(m, dtype=np.float32)
    bad[0] = -0.1
    bad[1] = 1.1

    monkeypatch.setattr(agent, "ask_for_participation", lambda *, area: True)
    monkeypatch.setattr(agent, "vote", lambda *, area: bad)

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        area._tally_votes()
