from __future__ import annotations

import numpy as np
import pytest

from tests.factory import create_test_model


pytestmark = pytest.mark.phase1


def _model_one_area(**overrides):
    base = dict(
        seed=301,
        num_agents=2,
        num_colors=3,
        num_areas=1,
        height=8,
        width=8,
        av_area_height=8,
        av_area_width=8,
        area_size_variance=0.0,
        max_steps=1,
        known_cells=0,
        mu=0.0,
        election_cost_rate=0.0,
        reward_rate_personal=0.0,
        altruism_mode="satisfaction",
        altruism_response_gamma=1.0,
        altruism_satisfaction_theta=0.5,
        altruism_satisfaction_slope=4.0,
        altruism_init=0.25,
        altruism_clip_min=0.0,
        altruism_clip_max=1.0,
    )
    base.update(overrides)
    model, _ = create_test_model(**base)
    return model


def test_satisfaction_mode_direct_mapping_gamma_1_uses_sigmoid_target() -> None:
    model = _model_one_area(
        altruism_response_gamma=1.0,
        altruism_init=0.1,
        altruism_satisfaction_theta=0.5,
        altruism_satisfaction_slope=4.0,
    )
    a = model.voting_agents[0]
    assert a is not None

    a.altruism_factor = 0.1
    a.apply_altruism_satisfaction_mode(dissatisfaction_value=0.2)
    s = 1.0 - 0.2
    expected = 1.0 / (1.0 + np.exp(-4.0 * (s - 0.5)))
    assert float(a.altruism_factor) == pytest.approx(float(expected), abs=1e-12)


def test_satisfaction_mode_gamma_smooths_toward_sigmoid_target() -> None:
    model = _model_one_area(
        altruism_response_gamma=0.25,
        altruism_init=0.2,
        altruism_satisfaction_theta=0.5,
        altruism_satisfaction_slope=4.0,
    )
    a = model.voting_agents[0]
    assert a is not None

    a.altruism_factor = 0.2
    a.apply_altruism_satisfaction_mode(dissatisfaction_value=0.0)
    target = 1.0 / (1.0 + np.exp(-4.0 * (1.0 - 0.5)))
    expected = (1.0 - 0.25) * 0.2 + 0.25 * float(target)
    assert float(a.altruism_factor) == pytest.approx(expected, abs=1e-12)


def test_satisfaction_mode_theta_and_slope_control_target() -> None:
    model = _model_one_area(
        altruism_response_gamma=1.0,
        altruism_init=0.1,
        altruism_satisfaction_theta=0.7,
        altruism_satisfaction_slope=12.0,
    )
    a = model.voting_agents[0]
    assert a is not None

    # same dissatisfaction, but high threshold should keep altruism low.
    a.apply_altruism_satisfaction_mode(dissatisfaction_value=0.4)  # s=0.6 < theta=0.7
    low_target = float(a.altruism_factor)
    assert low_target < 0.5

    a.apply_altruism_satisfaction_mode(dissatisfaction_value=0.2)  # s=0.8 > theta=0.7
    high_target = float(a.altruism_factor)
    assert high_target > 0.5
    assert high_target > low_target


def test_satisfaction_mode_updates_all_agents_pre_election_even_if_abstaining() -> None:
    model = _model_one_area(altruism_response_gamma=1.0, altruism_init=0.1, max_steps=1)

    class _NeverParticipate:
        def decide_participation(self, agent, area) -> bool:  # type: ignore[no-untyped-def]
            return False

    class _ZeroBallot:
        def score_options(self, agent, area, options) -> np.ndarray:  # type: ignore[no-untyped-def]
            return np.zeros(int(options.shape[0]), dtype=np.float32)

    for a in model.voting_agents:
        if a is None:
            continue
        a.participation_strategy = _NeverParticipate()
        a.voting_strategy = _ZeroBallot()

        def _sv(_self, *, area, model) -> float:  # type: ignore[no-untyped-def]
            return 0.3

        a.compute_dissatisfaction_value = _sv.__get__(a, type(a))

    model.step()

    vals = [float(a.altruism_factor) for a in model.voting_agents if a is not None]
    assert vals
    # target = sigmoid(4*(0.7-0.5)) ~= 0.68997, and this should apply even though nobody participated
    expected = 1.0 / (1.0 + np.exp(-4.0 * (0.7 - 0.5)))
    assert all(abs(v - float(expected)) < 1e-12 for v in vals)
