from __future__ import annotations

import numpy as np
import pytest

from tests.factory import create_test_model
from src.utils.representations import validate_ordering


pytestmark = pytest.mark.phase1


class _AlwaysParticipate:
    def decide_participation(self, agent, area) -> bool:  # type: ignore[no-untyped-def]
        return True


class _ZeroBallot:
    def score_options(self, agent, area, options) -> np.ndarray:  # type: ignore[no-untyped-def]
        # Valid ScoreVector in [0,1].
        return np.zeros(int(options.shape[0]), dtype=np.float32)


def _model_one_area(**overrides):
    base = dict(
        seed=9001,
        num_colors=3,
        num_agents=6,
        num_areas=1,
        height=8,
        width=8,
        av_area_height=8,
        av_area_width=8,
        area_size_variance=0.0,
        num_personality_groups=3,
        max_steps=1,
        election_cost_rate=0.0,
        reward_rate_common=0.0,
        reward_rate_personal=0.2,
        break_even_distance_common=0.5,
        break_even_distance_personal=0.5,
        abstention_share=1.0,
    )
    base.update(overrides)
    model, _ = create_test_model(**base)
    return model


def _force_all_participate_and_vote(model) -> None:
    for a in model.voting_agents:
        if a is None:
            continue
        a.participation_strategy = _AlwaysParticipate()
        a.voting_strategy = _ZeroBallot()


def test_break_even_distance_personal_monotone_linear_in_break_even(monkeypatch) -> None:
    """Metamorphic test: increasing break_even_distance_personal increases personal reward linearly.

    We patch distance_func to return a fixed personal distance p for all agents,
    so only the break_even changes the coefficient.
    """
    p = 0.6
    rate = 0.3
    t1 = 0.2
    t2 = 0.9
    assert t2 > t1

    m1 = _model_one_area(reward_rate_personal=rate, break_even_distance_personal=t1)
    m2 = _model_one_area(reward_rate_personal=rate, break_even_distance_personal=t2)

    _force_all_participate_and_vote(m1)
    _force_all_participate_and_vote(m2)

    # Patch both models: distance is constant regardless of inputs.
    monkeypatch.setattr(m1, "distance_func", lambda *_args, **_kw: float(p))
    monkeypatch.setattr(m2, "distance_func", lambda *_args, **_kw: float(p))

    area1 = m1.areas[0]
    area2 = m2.areas[0]
    assets1 = [float(a.assets) for a in area1.agents]
    assets2 = [float(a.assets) for a in area2.agents]
    assert assets1 == pytest.approx(assets2, abs=0.0)  # identical init under same seed

    m1.step()
    m2.step()

    # For each agent, reward difference equals (t2 - t1) * rate * assets_pre.
    for a0, a_t1, a_t2 in zip(assets1, area1.agents, area2.agents):
        r1 = float(getattr(a_t1, "_reward_pers_comp"))
        r2 = float(getattr(a_t2, "_reward_pers_comp"))
        expected_delta = (t2 - t1) * rate * a0
        assert (r2 - r1) == pytest.approx(expected_delta, abs=1e-9)


def test_break_even_distance_personal_produces_mixed_signs_across_agents(monkeypatch) -> None:
    """Semantic test: with p_low < break_even < p_high, some agents rewarded and others penalized."""
    model = _model_one_area(reward_rate_personal=0.5, break_even_distance_personal=0.5)
    _force_all_participate_and_vote(model)
    area = model.areas[0]
    assert len(area.agents) >= 2

    # Force two distinct personality group orderings so we can key off them.
    a0 = area.agents[0]
    a1 = area.agents[1]
    a0.personality_group = np.asarray([0, 1, 2], dtype=np.int64)
    a1.personality_group = np.asarray([2, 1, 0], dtype=np.int64)
    validate_ordering(a0.personality_group, int(model.num_colors))
    validate_ordering(a1.personality_group, int(model.num_colors))

    # Patch distance: return low value for a0, high for others.
    def _dist(x, _y, _pairs):  # type: ignore[no-untyped-def]
        xx = np.asarray(x)
        if np.array_equal(xx, np.asarray(a0.personality_group)):
            return 0.2  # below break-even -> reward
        return 0.8  # above break-even -> penalty

    monkeypatch.setattr(model, "distance_func", _dist)

    model.step()

    assert float(getattr(a0, "_reward_pers_comp")) > 0.0
    assert float(getattr(a1, "_reward_pers_comp")) < 0.0
