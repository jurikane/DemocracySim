from __future__ import annotations

import numpy as np
import pytest

from tests.factory import create_test_model
from src.utils.representations import validate_ordering, distribution_to_ordering


pytestmark = pytest.mark.phase1


class _AlwaysParticipate:
    def decide_participation(self, agent, area) -> bool:  # type: ignore[no-untyped-def]
        return True


class _ZeroBallot:
    def score_options(self, agent, area, options) -> np.ndarray:  # type: ignore[no-untyped-def]
        return np.zeros(int(options.shape[0]), dtype=np.float32)


def _model_one_area(**overrides):
    base = dict(
        seed=24,
        num_colors=3,
        num_agents=10,
        num_areas=1,
        height=8,
        width=8,
        av_area_height=8,
        av_area_width=8,
        area_size_variance=0.0,
        max_steps=1,
        election_cost_rate=0.0,
        reward_rate_common=0.0,
        break_even_distance_personal=0.5,
        break_even_distance_common=0.5,
        abstention_share=1.0,
    )
    base.update(overrides)
    model, _ = create_test_model(**base)
    return model


def _force_outcome_equals_first_agent_personality(model, area) -> None:
    """Elect the first agent's personality_group ordering (so personal distance=0 for that agent)."""
    a0 = area.agents[0]
    target = np.asarray(a0.personality_group, dtype=np.int64)
    validate_ordering(target, int(model.num_colors))

    opts = np.asarray(model.options)
    matches = np.nonzero((opts == target).all(axis=1))[0]
    assert len(matches) == 1
    win = int(matches[0])

    m = int(opts.shape[0])
    rest = [i for i in range(m) if i != win]
    ordering = np.asarray([win] + rest, dtype=np.int64)
    validate_ordering(ordering, m)

    def _rule(_pref_table: np.ndarray, *, rng) -> np.ndarray:  # type: ignore[no-untyped-def]
        return ordering

    model.voting_rule = _rule


def test_reward_rate_personal_zero_yields_no_personal_rewards() -> None:
    model = _model_one_area(reward_rate_personal=0.0, break_even_distance_personal=1.0)
    area = model.areas[0]
    # Avoid ties in "real ordering" paths (not critical here, but keeps determinism).
    area._color_distribution = np.asarray([0.1, 0.3, 0.6], dtype=np.float64)

    _force_outcome_equals_first_agent_personality(model, area)
    for a in model.voting_agents:
        if a is None:
            continue
        a.participation_strategy = _AlwaysParticipate()
        a.voting_strategy = _ZeroBallot()

    assets0 = [float(a.assets) for a in area.agents]
    model.step()

    assert [float(a.assets) for a in area.agents] == pytest.approx(assets0, abs=1e-12)
    for a in area.agents:
        assert float(getattr(a, "_reward_pers_comp")) == pytest.approx(0.0, abs=1e-12)


def test_reward_rate_personal_scales_personal_component_linearly_for_matching_agent() -> None:
    rate = 0.2
    thresh = 0.5
    model = _model_one_area(reward_rate_personal=rate, break_even_distance_personal=thresh)
    area = model.areas[0]
    area._color_distribution = np.asarray([0.1, 0.3, 0.6], dtype=np.float64)

    _force_outcome_equals_first_agent_personality(model, area)
    for a in model.voting_agents:
        if a is None:
            continue
        a.participation_strategy = _AlwaysParticipate()
        a.voting_strategy = _ZeroBallot()

    a0 = area.agents[0]
    assets0 = float(a0.assets)
    model.step()

    # For agent 0: p = dist(personality_group, voted_ordering) = 0
    # pers_coeff = break_even_personal - 0 = break_even_personal
    # pers_component = pers_coeff * reward_rate_personal * assets_pre
    expected_pers = thresh * rate * assets0
    assert float(getattr(a0, "_reward_pers_comp")) == pytest.approx(expected_pers, abs=1e-9)
    assert float(a0.assets) == pytest.approx(assets0 + expected_pers, abs=1e-9)


def test_reward_rate_personal_metamorphic_ratio_two_runs() -> None:
    """Metamorphic/property test: scaling reward_rate_personal scales personal rewards proportionally."""
    rate1 = 0.05
    rate2 = 0.2
    assert rate2 > rate1
    ratio = rate2 / rate1
    be = 0.6

    m1 = _model_one_area(seed=2026, reward_rate_personal=rate1, break_even_distance_personal=be)
    m2 = _model_one_area(seed=2026, reward_rate_personal=rate2, break_even_distance_personal=be)
    a1 = m1.areas[0]
    a2 = m2.areas[0]
    a1._color_distribution = np.asarray([0.1, 0.3, 0.6], dtype=np.float64)
    a2._color_distribution = np.asarray([0.1, 0.3, 0.6], dtype=np.float64)

    _force_outcome_equals_first_agent_personality(m1, a1)
    _force_outcome_equals_first_agent_personality(m2, a2)
    for ag in m1.voting_agents:
        if ag is None:
            continue
        ag.participation_strategy = _AlwaysParticipate()
        ag.voting_strategy = _ZeroBallot()
    for ag in m2.voting_agents:
        if ag is None:
            continue
        ag.participation_strategy = _AlwaysParticipate()
        ag.voting_strategy = _ZeroBallot()

    g1 = a1.agents[0]
    g2 = a2.agents[0]
    assets_pre_1 = float(g1.assets)
    assets_pre_2 = float(g2.assets)
    assert assets_pre_1 == pytest.approx(assets_pre_2, abs=0.0)

    m1.step()
    m2.step()

    r1 = float(getattr(g1, "_reward_pers_comp"))
    r2 = float(getattr(g2, "_reward_pers_comp"))
    assert (r2 / r1) == pytest.approx(ratio, rel=1e-10, abs=1e-12)


def test_reward_rate_personal_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        _model_one_area(reward_rate_personal=-0.1)
    with pytest.raises(ValueError):
        _model_one_area(reward_rate_personal=1.1)
