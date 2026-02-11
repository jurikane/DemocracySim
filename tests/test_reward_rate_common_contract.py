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
        # Valid ScoreVector in [0,1].
        return np.zeros(int(options.shape[0]), dtype=np.float32)


def _model_one_area(**overrides):
    base = dict(
        seed=42,
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
        # Keep threshold in-range; tests override as needed.
        reward_threshold_common=0.5,
    )
    base.update(overrides)
    model, _ = create_test_model(**base)
    return model


def _force_perfect_outcome_voting_rule(model, area) -> None:
    """Set model.voting_rule to always elect the 'real' ordering implied by area.color_distribution."""
    real = distribution_to_ordering(area.color_distribution, rng=model.voting_rng)
    validate_ordering(real, int(model.num_colors))

    opts = np.asarray(model.options)
    matches = np.nonzero((opts == np.asarray(real)).all(axis=1))[0]
    assert len(matches) == 1
    win = int(matches[0])

    m = int(opts.shape[0])
    rest = [i for i in range(m) if i != win]
    ordering = np.asarray([win] + rest, dtype=np.int64)
    validate_ordering(ordering, m)

    def _rule(_pref_table: np.ndarray, *, rng) -> np.ndarray:  # type: ignore[no-untyped-def]
        return ordering

    model.voting_rule = _rule


def test_reward_rate_common_zero_yields_no_common_rewards_and_no_asset_change() -> None:
    model = _model_one_area(reward_rate_common=0.0, reward_threshold_common=1.0)
    area = model.areas[0]
    # Avoid ties to keep the implied "real ordering" deterministic.
    area._color_distribution = np.asarray([0.1, 0.3, 0.6], dtype=np.float64)

    _force_perfect_outcome_voting_rule(model, area)
    for a in model.voting_agents:
        if a is None:
            continue
        a.participation_strategy = _AlwaysParticipate()
        a.voting_strategy = _ZeroBallot()

    assets0 = [float(a.assets) for a in area.agents]
    model.step()

    # No fees, no personal rewards, common rate is 0 => no asset change.
    assert [float(a.assets) for a in area.agents] == pytest.approx(assets0, abs=1e-12)
    for a in area.agents:
        assert float(getattr(a, "_reward_common_comp")) == pytest.approx(0.0, abs=1e-12)


def test_reward_rate_common_scales_common_component_linearly() -> None:
    rate = 0.2
    thresh = 0.5
    model = _model_one_area(reward_rate_common=rate, reward_threshold_common=thresh)
    area = model.areas[0]
    area._color_distribution = np.asarray([0.1, 0.3, 0.6], dtype=np.float64)

    _force_perfect_outcome_voting_rule(model, area)
    for a in model.voting_agents:
        if a is None:
            continue
        a.participation_strategy = _AlwaysParticipate()
        a.voting_strategy = _ZeroBallot()

    assets0 = [float(a.assets) for a in area.agents]
    model.step()

    # Perfect outcome => dist_to_reality == 0 => common_coeff == threshold_common.
    # common_component = threshold_common * reward_rate_common * assets_pre
    for a0, a in zip(assets0, area.agents):
        expected_common = thresh * rate * a0
        assert float(getattr(a, "_reward_common_comp")) == pytest.approx(expected_common, abs=1e-9)
        assert float(a.assets) == pytest.approx(a0 + expected_common, abs=1e-9)


def test_reward_rate_common_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        _model_one_area(reward_rate_common=-0.1)
    with pytest.raises(ValueError):
        _model_one_area(reward_rate_common=1.1)
