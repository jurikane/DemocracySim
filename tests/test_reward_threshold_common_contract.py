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
        seed=101,
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
        reward_rate_personal=0.0,
        reward_rate_common=0.2,
        abstention_share=1.0,
    )
    base.update(overrides)
    model, _ = create_test_model(**base)
    return model


def _force_perfect_outcome(model, area) -> None:
    real = distribution_to_ordering(area.color_distribution, rng=model.voting_rng)
    validate_ordering(real, int(model.num_colors))
    opts = np.asarray(model.options)
    matches = np.nonzero((opts == np.asarray(real)).all(axis=1))[0]
    assert len(matches) == 1
    win = int(matches[0])
    m = int(opts.shape[0])
    rest = [i for i in range(m) if i != win]
    ordering = np.asarray([win] + rest, dtype=np.int64)

    def _rule(_pref_table: np.ndarray, *, rng) -> np.ndarray:  # type: ignore[no-untyped-def]
        return ordering

    model.voting_rule = _rule


def test_break_even_distance_common_controls_sign_and_magnitude_at_dist_zero() -> None:
    rate = 0.2

    # Case A: break_even=0 => common_coeff = 0 => no common rewards
    model0 = _model_one_area(reward_rate_common=rate, break_even_distance_common=0.0)
    area0 = model0.areas[0]
    area0._color_distribution = np.asarray([0.1, 0.3, 0.6], dtype=np.float64)
    _force_perfect_outcome(model0, area0)
    for a in model0.voting_agents:
        if a is None:
            continue
        a.participation_strategy = _AlwaysParticipate()
        a.voting_strategy = _ZeroBallot()
    assets0 = [float(a.assets) for a in area0.agents]
    model0.step()
    assert [float(a.assets) for a in area0.agents] == pytest.approx(assets0, abs=1e-12)

    # Case B: break_even=1 => common_coeff = 1 => maximal common reward
    model1 = _model_one_area(reward_rate_common=rate, break_even_distance_common=1.0)
    area1 = model1.areas[0]
    area1._color_distribution = np.asarray([0.1, 0.3, 0.6], dtype=np.float64)
    _force_perfect_outcome(model1, area1)
    for a in model1.voting_agents:
        if a is None:
            continue
        a.participation_strategy = _AlwaysParticipate()
        a.voting_strategy = _ZeroBallot()
    assets1 = [float(a.assets) for a in area1.agents]
    model1.step()
    for a0, a in zip(assets1, area1.agents):
        expected = 1.0 * rate * a0
        assert float(getattr(a, "_reward_common_comp")) == pytest.approx(expected, abs=1e-9)
        assert float(a.assets) == pytest.approx(a0 + expected, abs=1e-9)


def test_break_even_distance_common_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        _model_one_area(break_even_distance_common=-0.01)
    with pytest.raises(ValueError):
        _model_one_area(break_even_distance_common=1.01)

