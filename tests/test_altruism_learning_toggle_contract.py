from __future__ import annotations

import numpy as np
import pytest

from tests.factory import create_test_model
from src.agents.strategies import DefaultParticipationStrategy


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


def _force_fixed_voting_rule(model) -> None:
    """Deterministic rule: choose option 0 as winner, then ascending order."""
    m = int(np.asarray(model.options).shape[0])
    ordering = np.arange(m, dtype=np.int64)

    def _rule(_pref_table: np.ndarray, *, rng) -> np.ndarray:  # type: ignore[no-untyped-def]
        return ordering

    model.voting_rule = _rule


def test_altruism_learning_oracle_initialization_static_vs_learning() -> None:
    m_static, _ = create_test_model(
        seed=60,
        num_agents=1,
        num_colors=3,
        num_areas=1,
        altruism_learning=False,
        altruism_static=0.25,
        altruism_init=0.9,
    )
    a_s = m_static.voting_agents[0]
    assert a_s is not None
    assert float(a_s.altruism_factor) == pytest.approx(0.25, abs=0.0)

    m_learn, _ = create_test_model(
        seed=60,
        num_agents=1,
        num_colors=3,
        num_areas=1,
        altruism_learning=True,
        altruism_static=0.25,
        altruism_init=0.9,
    )
    a_l = m_learn.voting_agents[0]
    assert a_l is not None
    assert float(a_l.altruism_factor) == pytest.approx(0.9, abs=0.0)


def test_altruism_learning_metamorphic_toggle_gates_update() -> None:
    """Metamorphic: with identical state and dissatisfaction_signal, toggling altruism_learning
    determines whether altruism_factor changes during conduct_election().
    """
    cfg = dict(
        seed=61,
        num_agents=2,
        num_colors=3,
        num_areas=1,
        height=8,
        width=8,
        av_area_height=8,
        av_area_width=8,
        area_size_variance=0.0,
        max_steps=1,
        election_cost_rate=0.0,
        reward_rate_common=0.0,
        reward_rate_personal=0.0,
        altruism_alpha=0.2,
        altruism_init=0.5,
        altruism_clip_min=0.0,
        altruism_clip_max=1.0,
    )
    m_off, _ = create_test_model(**cfg, altruism_learning=False, altruism_static=0.5)
    m_on, _ = create_test_model(**cfg, altruism_learning=True, altruism_static=0.5)

    a_off = m_off.areas[0]
    a_on = m_on.areas[0]
    # Avoid ties in the "real ordering" computation.
    a_off._color_distribution = np.asarray([0.1, 0.3, 0.6], dtype=np.float64)
    a_on._color_distribution = np.asarray([0.1, 0.3, 0.6], dtype=np.float64)

    _force_fixed_voting_rule(m_off)
    _force_fixed_voting_rule(m_on)

    # Force 1 participant, 1 abstainer.
    off_part = a_off.agents[0]
    off_abs = a_off.agents[1]
    on_part = a_on.agents[0]
    on_abs = a_on.agents[1]
    for ag in m_off.voting_agents:
        if ag is None:
            continue
        ag.participation_strategy = _NeverParticipate()
        ag.voting_strategy = _ZeroBallot()
    for ag in m_on.voting_agents:
        if ag is None:
            continue
        ag.participation_strategy = _NeverParticipate()
        ag.voting_strategy = _ZeroBallot()
    off_part.participation_strategy = _AlwaysParticipate()
    on_part.participation_strategy = _AlwaysParticipate()

    # Inject a known signal; conduct_election() will pass this through to altruism update
    # only if altruism_learning is enabled, and only for participating agents.
    for ag in [off_part, off_abs, on_part, on_abs]:
        ag.dissatisfaction_signal = 0.5

    off_part_a0 = float(off_part.altruism_factor)
    off_abs_a0 = float(off_abs.altruism_factor)
    on_part_a0 = float(on_part.altruism_factor)
    on_abs_a0 = float(on_abs.altruism_factor)

    a_off.conduct_election()
    a_on.conduct_election()

    # Learning off: no updates.
    assert float(off_part.altruism_factor) == pytest.approx(off_part_a0, abs=0.0)
    assert float(off_abs.altruism_factor) == pytest.approx(off_abs_a0, abs=0.0)

    # Learning on: only participant updates.
    expected = on_part_a0 + float(m_on.altruism_alpha) * 0.5
    assert float(on_part.altruism_factor) == pytest.approx(expected, abs=1e-12)
    assert float(on_abs.altruism_factor) == pytest.approx(on_abs_a0, abs=0.0)
