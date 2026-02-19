from __future__ import annotations

import numpy as np
import pytest

from tests.factory import create_test_model
from src.agents.strategies import DefaultParticipationStrategy


pytestmark = pytest.mark.phase1


class _AlwaysParticipate:
    def decide_participation(self, agent, area) -> bool:  # type: ignore[no-untyped-def]
        return True


class _SequenceParticipate:
    """Deterministic participation sequence for one agent."""

    def __init__(self, seq: list[bool]):
        self._it = iter([bool(x) for x in seq])

    def decide_participation(self, agent, area) -> bool:  # type: ignore[no-untyped-def]
        return bool(next(self._it))


class _ZeroBallot:
    def score_options(self, agent, area, options) -> np.ndarray:  # type: ignore[no-untyped-def]
        return np.zeros(int(options.shape[0]), dtype=np.float32)


def _force_fixed_voting_rule(model) -> None:
    """Deterministic rule: pick option 0, then ascending order of the rest."""
    m = int(np.asarray(model.options).shape[0])
    ordering = np.arange(m, dtype=np.int64)

    def _rule(_pref_table: np.ndarray, *, rng) -> np.ndarray:  # type: ignore[no-untyped-def]
        return ordering

    model.voting_rule = _rule


class _SeqNpRandom:
    """Wrapper around a NumPy Generator that yields a deterministic random() sequence.

    We delegate .choice() to the underlying generator so other model code keeps working.
    """

    def __init__(self, base, draws: list[float]):
        self._base = base
        self._it = iter([float(x) for x in draws])

    def random(self) -> float:
        return float(next(self._it))

    def choice(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return self._base.choice(*args, **kwargs)


def _make_two_agent_model_for_participation_learning(**overrides):
    base = dict(
        seed=700,
        num_agents=2,
        num_colors=3,
        num_areas=1,
        height=8,
        width=8,
        av_area_height=8,
        av_area_width=8,
        area_size_variance=0.0,
        known_cells=0,  # avoid np_random.choice in knowledge sampling
        mu=0.0,  # avoid mutation effects
        max_steps=10,
        # isolate participation learning by default
        reward_rate_common=0.0,
        reward_rate_personal=0.0,
        break_even_distance_common=0.5,
        break_even_distance_personal=0.5,
        abstention_share=0.0,
    )
    base.update(overrides)
    model, _ = create_test_model(**base)
    _force_fixed_voting_rule(model)
    # Ensure deterministic ballots.
    for a in model.voting_agents:
        if a is None:
            continue
        a.voting_strategy = _ZeroBallot()
    # Keep agent 1 always participating so the election always runs.
    model.voting_agents[1].participation_strategy = _AlwaysParticipate()  # type: ignore[union-attr]
    return model


def test_participation_learning_interaction_baseline_alpha_controls_persistence() -> None:
    """V1 contract: learning uses level signal (signal == delta_rel).

    Consequence: participation_baseline_alpha must not affect q-updates.
    """
    rate = 0.4
    alpha_q = 1.0
    init_q = 0.0

    m_fast, _ = create_test_model(
        seed=701,
        num_agents=2,
        num_colors=3,
        num_areas=1,
        height=8,
        width=8,
        av_area_height=8,
        av_area_width=8,
        area_size_variance=0.0,
        known_cells=0,
        mu=0.0,
        max_steps=3,
        election_cost_rate=rate,
        reward_rate_common=0.0,
        reward_rate_personal=0.0,
        participation_alpha=alpha_q,
        participation_init_q=init_q,
        participation_q_max=1_000.0,
        participation_baseline_alpha=1.0,
    )
    m_slow, _ = create_test_model(
        seed=701,
        num_agents=2,
        num_colors=3,
        num_areas=1,
        height=8,
        width=8,
        av_area_height=8,
        av_area_width=8,
        area_size_variance=0.0,
        known_cells=0,
        mu=0.0,
        max_steps=3,
        election_cost_rate=rate,
        reward_rate_common=0.0,
        reward_rate_personal=0.0,
        participation_alpha=alpha_q,
        participation_init_q=init_q,
        participation_q_max=1_000.0,
        participation_baseline_alpha=0.0,
    )
    _force_fixed_voting_rule(m_fast)
    _force_fixed_voting_rule(m_slow)
    for m in (m_fast, m_slow):
        for a in m.voting_agents:
            if a is None:
                continue
            a.voting_strategy = _ZeroBallot()
        m.voting_agents[1].participation_strategy = _AlwaysParticipate()  # type: ignore[union-attr]

    learner_fast = m_fast.voting_agents[0]
    learner_slow = m_slow.voting_agents[0]
    assert learner_fast is not None and learner_slow is not None
    # Fixed action path for learner:
    # step1 participate (delta_rel=-rate), then abstain twice (delta_rel=0).
    learner_fast.participation_strategy = _SequenceParticipate([True, False, False])
    learner_slow.participation_strategy = _SequenceParticipate([True, False, False])

    # Run 3 steps and capture level-signal contract.
    traces_fast: list[tuple[float, float]] = []
    traces_slow: list[tuple[float, float]] = []
    for _ in range(3):
        m_fast.step()
        m_slow.step()
        traces_fast.append((float(learner_fast.participation_signal), float(learner_fast.election_delta_rel)))
        traces_slow.append((float(learner_slow.participation_signal), float(learner_slow.election_delta_rel)))

    # Learning signal must equal realized delta_rel step-wise.
    for sig, drel in traces_fast + traces_slow:
        assert sig == pytest.approx(drel, abs=1e-12)

    # q update path is baseline-alpha invariant under V1:
    # step1: q += alpha_q * (+1) * (-rate) = -rate
    # step2/3: delta_rel=0 => no further change
    q_expected = init_q - alpha_q * rate
    assert float(learner_fast.q_participation) == pytest.approx(q_expected, abs=1e-12)
    assert float(learner_slow.q_participation) == pytest.approx(q_expected, abs=1e-12)


def test_participation_learning_interaction_q_max_clips_under_repeated_surprise() -> None:
    """Interaction: repeated negative level outcomes are clipped by q_max."""
    rate = 0.2
    alpha_q = 1.0
    q_max = 0.3
    init_q = 0.0

    model = _make_two_agent_model_for_participation_learning(
        seed=702,
        max_steps=6,
        election_cost_rate=rate,
        participation_alpha=alpha_q,
        participation_init_q=init_q,
        participation_q_max=q_max,
        participation_baseline_alpha=0.0,
    )

    learner = model.voting_agents[0]
    assert learner is not None
    learner.participation_strategy = _AlwaysParticipate()

    # With level signal and always-participate:
    # each step contributes q += alpha_q * (+1) * (-rate) = -rate.
    # Over 6 steps this would be -1.2, so clipping at -q_max must activate.

    for _ in range(6):
        model.step()

    assert float(learner.q_participation) == pytest.approx(-q_max, abs=1e-12)


def test_participation_learning_interaction_beta_amplifies_effect_of_q_change_on_decision() -> None:
    """Interaction: same learned q, different beta -> different participation decision."""
    q_fixed = 0.4
    b_lo = 0.5
    b_hi = 5.0
    u = 0.7  # between sigmoid(0.5*0.4) and sigmoid(5.0*0.4)

    def _mk(beta: float):
        m = _make_two_agent_model_for_participation_learning(
            seed=703,
            max_steps=1,
            participation_q_max=1_000.0,
            participation_beta=beta,
            participation_baseline_alpha=1.0,
        )
        m.participation_rng = _SeqNpRandom(m.participation_rng, [u])
        learner = m.voting_agents[0]
        assert learner is not None
        learner.participation_strategy = DefaultParticipationStrategy()
        learner.q_participation = q_fixed
        return m, learner

    m_lo, a_lo = _mk(b_lo)
    m_hi, a_hi = _mk(b_hi)

    # Decision differs by beta given same q and same RNG draw.
    m_lo.step()
    m_hi.step()

    # With low beta: p ~= sigmoid(0.2) ~ 0.55 => u=0.7 => abstain.
    assert bool(a_lo.participating) is False
    # With high beta: p ~= sigmoid(2.0) ~ 0.88 => u=0.7 => participate.
    assert bool(a_hi.participating) is True
