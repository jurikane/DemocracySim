from __future__ import annotations

import numpy as np
import pytest

from tests.factory import create_test_model
from src.utils.representations import validate_ordering


pytestmark = pytest.mark.phase1


class _AlwaysParticipate:
    def decide_participation(self, agent, area) -> bool:  # type: ignore[no-untyped-def]
        return True


class _NeverParticipate:
    def decide_participation(self, agent, area) -> bool:  # type: ignore[no-untyped-def]
        return False


class _ZeroBallot:
    def score_options(self, agent, area, options) -> np.ndarray:  # type: ignore[no-untyped-def]
        return np.zeros(int(options.shape[0]), dtype=np.float32)


def _model_one_area(**overrides):
    base = dict(
        seed=77,
        num_colors=3,
        num_agents=12,
        num_areas=1,
        height=8,
        width=8,
        av_area_height=8,
        av_area_width=8,
        area_size_variance=0.0,
        max_steps=1,
        election_cost_rate=0.0,
        reward_rate_personal=0.1,
        break_even_distance_common=0.5,
    )
    base.update(overrides)
    model, _ = create_test_model(**base)
    return model


def _force_winning_ordering(model, ordering: np.ndarray) -> None:
    opts = np.asarray(model.options)
    ord_arr = np.asarray(ordering, dtype=np.int64)
    validate_ordering(ord_arr, int(model.num_colors))
    matches = np.nonzero((opts == ord_arr).all(axis=1))[0]
    assert len(matches) == 1
    win = int(matches[0])
    rest = [i for i in range(int(opts.shape[0])) if i != win]
    out = np.asarray([win] + rest, dtype=np.int64)

    def _rule(_pref_table: np.ndarray, *, rng) -> np.ndarray:  # type: ignore[no-untyped-def]
        return out

    model.voting_rule = _rule


def _prepare_all_agents_vote(model) -> None:
    for a in model.voting_agents:
        if a is None:
            continue
        a.participation_strategy = _AlwaysParticipate()
        a.voting_strategy = _ZeroBallot()


def test_binary_quality_good_decision_reward_formula_oracle() -> None:
    model = _model_one_area(
        reward_rate_personal=0.10,
        break_even_distance_common=1.0,  # always "good" since dist_to_reality in [0,1]
        election_cost_rate=0.0,
    )
    area = model.areas[0]
    area._color_distribution = np.asarray([0.6, 0.3, 0.1], dtype=np.float64)

    # Pick a concrete elected ordering from an existing agent to avoid option mismatch.
    elected = np.asarray(area.agents[0].personality_group, dtype=np.int64)
    _force_winning_ordering(model, elected)
    _prepare_all_agents_vote(model)

    assets_pre = {a.unique_id: float(a.assets) for a in area.agents}
    model.step()

    rate = float(model.reward_rate_personal)
    dist_func = model.distance_func
    pairs = model.color_search_pairs
    for a in area.agents:
        group_dst_to_outcome = float(dist_func(a.personality_group, elected, pairs))
        expected_reward = (1.0 - group_dst_to_outcome) * rate * assets_pre[a.unique_id]
        assert float(a.reward_common_component) == pytest.approx(0.0, abs=1e-12)
        assert float(a.reward_personal_component) == pytest.approx(expected_reward, abs=1e-9)
        assert float(a.election_delta_abs) == pytest.approx(expected_reward, abs=1e-9)


def test_binary_quality_bad_decision_reward_formula_oracle() -> None:
    model = _model_one_area(
        reward_rate_personal=0.10,
        break_even_distance_common=0.0,  # "bad" for any dist_to_reality > 0
        election_cost_rate=0.0,
    )
    area = model.areas[0]
    area._color_distribution = np.asarray([0.6, 0.3, 0.1], dtype=np.float64)

    elected = np.asarray([2, 1, 0], dtype=np.int64)  # opposite of real-order for the chosen distribution
    _force_winning_ordering(model, elected)
    _prepare_all_agents_vote(model)

    assets_pre = {a.unique_id: float(a.assets) for a in area.agents}
    model.step()

    assert float(area.dist_to_reality) > 0.0
    rate = float(model.reward_rate_personal)
    dist_func = model.distance_func
    pairs = model.color_search_pairs
    for a in area.agents:
        group_dst_to_outcome = float(dist_func(a.personality_group, elected, pairs))
        expected_reward = -(group_dst_to_outcome * rate * assets_pre[a.unique_id])
        assert float(a.reward_common_component) == pytest.approx(0.0, abs=1e-12)
        assert float(a.reward_personal_component) == pytest.approx(expected_reward, abs=1e-9)
        assert float(a.election_delta_abs) == pytest.approx(expected_reward, abs=1e-9)


def test_binary_quality_participation_fee_applies_only_to_participants() -> None:
    model = _model_one_area(
        reward_rate_personal=0.10,
        break_even_distance_common=1.0,
        election_cost_rate=0.05,
    )
    area = model.areas[0]
    area._color_distribution = np.asarray([0.6, 0.3, 0.1], dtype=np.float64)
    elected = np.asarray(area.agents[0].personality_group, dtype=np.int64)
    _force_winning_ordering(model, elected)

    # Exactly one participant to keep election valid and isolate fee effect.
    participant_id = area.agents[0].unique_id
    for a in area.agents:
        a.voting_strategy = _ZeroBallot()
        a.participation_strategy = (
            _AlwaysParticipate() if a.unique_id == participant_id else _NeverParticipate()
        )

    assets_pre = {a.unique_id: float(a.assets) for a in area.agents}
    model.step()

    rate = float(model.reward_rate_personal)
    dist_func = model.distance_func
    pairs = model.color_search_pairs
    for a in area.agents:
        group_dst_to_outcome = float(dist_func(a.personality_group, elected, pairs))
        expected_reward = (1.0 - group_dst_to_outcome) * rate * assets_pre[a.unique_id]
        expected_fee = float(model.election_cost_rate) * assets_pre[a.unique_id] if a.participating else 0.0
        expected_delta = expected_reward - expected_fee
        assert float(a.election_delta_abs) == pytest.approx(expected_delta, abs=1e-9)


def test_binary_quality_reward_rate_linear_metamorphic() -> None:
    r1 = 0.05
    r2 = 0.20
    ratio = r2 / r1
    m1 = _model_one_area(seed=1234, reward_rate_personal=r1, break_even_distance_common=1.0)
    m2 = _model_one_area(seed=1234, reward_rate_personal=r2, break_even_distance_common=1.0)
    a1 = m1.areas[0]
    a2 = m2.areas[0]
    a1._color_distribution = np.asarray([0.6, 0.3, 0.1], dtype=np.float64)
    a2._color_distribution = np.asarray([0.6, 0.3, 0.1], dtype=np.float64)
    elected = np.asarray(a1.agents[0].personality_group, dtype=np.int64)
    _force_winning_ordering(m1, elected)
    _force_winning_ordering(m2, elected)
    _prepare_all_agents_vote(m1)
    _prepare_all_agents_vote(m2)

    m1.step()
    m2.step()
    # Compare one agent with non-zero reward under good-decision branch.
    g1 = a1.agents[0]
    g2 = a2.agents[0]
    rwd1 = float(g1.reward_personal_component)
    rwd2 = float(g2.reward_personal_component)
    assert rwd1 > 0.0
    assert (rwd2 / rwd1) == pytest.approx(ratio, rel=1e-10, abs=1e-12)

