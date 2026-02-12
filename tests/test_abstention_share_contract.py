from __future__ import annotations

import numpy as np
import pytest

from tests.factory import create_test_model
from src.utils.representations import validate_ordering, distribution_to_ordering


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
        seed=555,
        num_colors=3,
        num_agents=6,
        num_areas=1,
        height=8,
        width=8,
        av_area_height=8,
        av_area_width=8,
        area_size_variance=0.0,
        max_steps=1,
        election_cost_rate=0.0,
        reward_rate_common=0.2,
        reward_rate_personal=0.0,
        break_even_distance_common=0.5,
        break_even_distance_personal=0.5,
        abstention_share=1.0,
    )
    base.update(overrides)
    model, _ = create_test_model(**base)
    return model


def _force_perfect_outcome(model, area) -> None:
    """Elect the real ordering so dist_to_reality==0 (common coeff == break_even_distance_common)."""
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


def test_abstention_share_applies_only_to_abstainers_oracle() -> None:
    """Oracle/semantic test with one participant and one abstainer.

    With perfect outcome (dist=0), the expected common component is:
      common = break_even_distance_common * reward_rate_common * assets_pre
    For abstainers: common *= abstention_share
    """
    share = 0.25
    model = _model_one_area(abstention_share=share, break_even_distance_common=0.5, reward_rate_common=0.2)
    area = model.areas[0]
    area._color_distribution = np.asarray([0.1, 0.3, 0.6], dtype=np.float64)
    _force_perfect_outcome(model, area)

    # Pick 2 agents to make the check crystal clear.
    a_part = area.agents[0]
    a_abst = area.agents[1]
    for a in model.voting_agents:
        if a is None:
            continue
        a.voting_strategy = _ZeroBallot()
        a.participation_strategy = _NeverParticipate()
    a_part.participation_strategy = _AlwaysParticipate()

    assets_part = float(a_part.assets)
    assets_abst = float(a_abst.assets)

    model.step()

    coeff = float(model.break_even_distance_common) - 0.0
    expected_part = coeff * float(model.reward_rate_common) * assets_part
    expected_abst = coeff * float(model.reward_rate_common) * assets_abst * share

    assert float(getattr(a_part, "_reward_common_comp")) == pytest.approx(expected_part, abs=1e-9)
    assert float(getattr(a_abst, "_reward_common_comp")) == pytest.approx(expected_abst, abs=1e-9)


def test_abstention_share_metamorphic_linear_scaling(monkeypatch) -> None:
    """Metamorphic: changing abstention_share scales abstainers' common reward linearly."""
    fixed_dist = 0.2
    rate = 0.3
    be = 0.6
    s1 = 0.1
    s2 = 0.9
    assert s2 > s1

    m1 = _model_one_area(abstention_share=s1, reward_rate_common=rate, break_even_distance_common=be)
    m2 = _model_one_area(abstention_share=s2, reward_rate_common=rate, break_even_distance_common=be)
    a1 = m1.areas[0]
    a2 = m2.areas[0]
    a1._color_distribution = np.asarray([0.1, 0.3, 0.6], dtype=np.float64)
    a2._color_distribution = np.asarray([0.1, 0.3, 0.6], dtype=np.float64)
    _force_perfect_outcome(m1, a1)
    _force_perfect_outcome(m2, a2)

    # Everyone abstains (still eligible), so no votes but reward logic still applies? In current model,
    # _distribute_rewards is not called when no one participates. So we need at least one participant.
    for a in m1.voting_agents:
        if a is None:
            continue
        a.voting_strategy = _ZeroBallot()
        a.participation_strategy = _NeverParticipate()
    for a in m2.voting_agents:
        if a is None:
            continue
        a.voting_strategy = _ZeroBallot()
        a.participation_strategy = _NeverParticipate()

    # Force exactly one participant, keep at least one abstainer.
    a_part_1 = a1.agents[0]
    a_abst_1 = a1.agents[1]
    a_part_2 = a2.agents[0]
    a_abst_2 = a2.agents[1]
    a_part_1.participation_strategy = _AlwaysParticipate()
    a_part_2.participation_strategy = _AlwaysParticipate()

    # Patch dist_to_reality to fixed value.
    monkeypatch.setattr(m1, "distance_func", lambda *_args, **_kw: float(fixed_dist))
    monkeypatch.setattr(m2, "distance_func", lambda *_args, **_kw: float(fixed_dist))

    assets_abst = float(a_abst_1.assets)
    assert assets_abst == pytest.approx(float(a_abst_2.assets), abs=0.0)

    m1.step()
    m2.step()

    r1 = float(getattr(a_abst_1, "_reward_common_comp"))
    r2 = float(getattr(a_abst_2, "_reward_common_comp"))
    coeff = be - fixed_dist
    base = coeff * rate * assets_abst
    assert r1 == pytest.approx(base * s1, abs=1e-9)
    assert r2 == pytest.approx(base * s2, abs=1e-9)


def test_abstention_share_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        _model_one_area(abstention_share=-0.1)
    with pytest.raises(ValueError):
        _model_one_area(abstention_share=1.1)
