from __future__ import annotations

import numpy as np
import pytest

from tests.factory import create_test_model


pytestmark = pytest.mark.phase1


class _AlwaysParticipate:
    def decide_participation(self, agent, area) -> bool:  # type: ignore[no-untyped-def]
        return True


class _ZeroBallot:
    def score_options(self, agent, area, options) -> np.ndarray:  # type: ignore[no-untyped-def]
        return np.zeros(int(options.shape[0]), dtype=np.float32)


def _model_one_area(**overrides):
    base = dict(
        seed=2020,
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
        abstention_share=1.0,
    )
    base.update(overrides)
    model, _ = create_test_model(**base)
    return model


def test_break_even_distance_common_is_break_even_distance(monkeypatch) -> None:
    """Lock semantics: common_coeff = break_even_common - dist_to_reality."""
    fixed_dist = 0.25
    rate = 0.2
    break_even = 0.1  # below fixed_dist => negative coeff (penalty)

    model = _model_one_area(
        reward_rate_common=rate,
        reward_rate_personal=0.0,
        break_even_distance_common=break_even,
        break_even_distance_personal=0.5,
    )
    area = model.areas[0]

    for a in model.voting_agents:
        if a is None:
            continue
        a.participation_strategy = _AlwaysParticipate()
        a.voting_strategy = _ZeroBallot()

    # Force dist_to_reality and personal distance to deterministic values without
    # depending on a particular distance function implementation.
    monkeypatch.setattr(model, "distance_func", lambda *_args, **_kw: float(fixed_dist))

    assets0 = [float(a.assets) for a in area.agents]
    model.step()

    coeff = break_even - fixed_dist
    for a0, a in zip(assets0, area.agents):
        expected = coeff * rate * a0
        assert float(getattr(a, "_reward_common_comp")) == pytest.approx(expected, abs=1e-9)


def test_break_even_distance_personal_is_break_even_distance(monkeypatch) -> None:
    """Lock semantics: pers_coeff = break_even_personal - distance(personality_group, voted_ordering)."""
    fixed_p = 0.6
    rate = 0.3
    break_even = 0.9  # above fixed_p => positive coeff (reward)

    model = _model_one_area(
        reward_rate_common=0.0,
        reward_rate_personal=rate,
        break_even_distance_common=0.5,
        break_even_distance_personal=break_even,
    )
    area = model.areas[0]

    for a in model.voting_agents:
        if a is None:
            continue
        a.participation_strategy = _AlwaysParticipate()
        a.voting_strategy = _ZeroBallot()

    monkeypatch.setattr(model, "distance_func", lambda *_args, **_kw: float(fixed_p))

    assets0 = [float(a.assets) for a in area.agents]
    model.step()

    coeff = break_even - fixed_p
    for a0, a in zip(assets0, area.agents):
        expected = coeff * rate * a0
        assert float(getattr(a, "_reward_pers_comp")) == pytest.approx(expected, abs=1e-9)

