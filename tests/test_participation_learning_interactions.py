from __future__ import annotations

import numpy as np
import pytest

from tests.factory import create_test_model
from src.agents.strategies import DefaultParticipationStrategy


pytestmark = pytest.mark.phase1


class _AlwaysParticipate:
    def decide_participation(self, agent, area) -> bool:  # type: ignore[no-untyped-def]
        return True


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
    """Interaction: participation_baseline_alpha controls how long a surprise persists.

    Scenario:
    - Step 1: learner participates => delta_rel = -election_cost_rate
      baseline initializes to delta1, signal=0 => no q update.
    - Steps 2-3: learner abstains => delta_rel = 0 each step
      signal stays positive while baseline remains negative.

    With baseline_alpha=1:
      baseline jumps to 0 after step2 => step3 signal=0 => only 1 q update total.
    With baseline_alpha=0:
      baseline stays at -rate => step2 and step3 signal=+rate => 2 q updates total.
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

    # Learner uses DefaultParticipationStrategy; drive its actions via fixed draws:
    # step1: participate, step2-3: abstain.
    draws = [0.0, 0.999, 0.999]
    m_fast.np_random = _SeqNpRandom(m_fast.np_random, draws)
    m_slow.np_random = _SeqNpRandom(m_slow.np_random, draws)

    learner_fast = m_fast.voting_agents[0]
    learner_slow = m_slow.voting_agents[0]
    assert learner_fast is not None and learner_slow is not None
    learner_fast.participation_strategy = DefaultParticipationStrategy()
    learner_slow.participation_strategy = DefaultParticipationStrategy()

    # Run 3 steps.
    for _ in range(3):
        m_fast.step()
        m_slow.step()

    # Expected q drift:
    # step2 update: q -= alpha_q * rate
    q2 = init_q - alpha_q * rate
    # step3 update magnitude depends on baseline alpha:
    # - baseline_alpha=1 => no update on step3
    # - baseline_alpha=0 => another -alpha_q*rate update
    assert float(learner_fast.q_participation) == pytest.approx(q2, abs=1e-12)
    assert float(learner_slow.q_participation) == pytest.approx(q2 - alpha_q * rate, abs=1e-12)


def test_participation_learning_interaction_q_max_clips_under_repeated_surprise() -> None:
    """Interaction: with persistent positive surprises, q drifts until clipped by q_max."""
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
        participation_baseline_alpha=0.0,  # keep baseline fixed at delta1
    )

    learner = model.voting_agents[0]
    assert learner is not None
    learner.participation_strategy = DefaultParticipationStrategy()

    # step1 participate => baseline=-rate, no update
    # steps2..6 abstain => signal=+rate every time => q decreases by alpha*rate each step until clipped.
    draws = [0.0] + [0.999] * 5
    model.np_random = _SeqNpRandom(model.np_random, draws)

    for _ in range(6):
        model.step()

    assert float(learner.q_participation) == pytest.approx(-q_max, abs=1e-12)


def test_participation_learning_interaction_beta_amplifies_effect_of_q_change_on_decision() -> None:
    """Interaction: for the same learned q increase, higher beta can flip the action.

    We create a 2-step sequence where the learner participates twice, and we reduce
    election_cost_rate to make step2 "better" than step1, yielding a positive signal
    and a positive q update for a participating agent.
    """
    cost1 = 0.4
    cost2 = 0.0
    alpha_q = 1.0
    init_q = 0.0
    b_lo = 0.5
    b_hi = 5.0
    u3 = 0.7  # between sigmoid(b_lo*q) and sigmoid(b_hi*q) for q ~= 0.4

    def _mk(beta: float):
        m = _make_two_agent_model_for_participation_learning(
            seed=703,
            max_steps=3,
            election_cost_rate=cost1,
            participation_alpha=alpha_q,
            participation_init_q=init_q,
            participation_q_max=1_000.0,
            participation_beta=beta,
            participation_baseline_alpha=1.0,
        )
        # Force learner to participate on steps 1-2 regardless of p by using u=0.
        m.np_random = _SeqNpRandom(m.np_random, [0.0, 0.0, u3])
        learner = m.voting_agents[0]
        assert learner is not None
        learner.participation_strategy = DefaultParticipationStrategy()
        return m, learner

    m_lo, a_lo = _mk(b_lo)
    m_hi, a_hi = _mk(b_hi)

    # Step 1: high cost => delta_rel = -0.4 (baseline init, signal 0).
    m_lo.step()
    m_hi.step()

    # Step 2: reduce cost => delta_rel becomes 0 (better than baseline by +0.4),
    # participating + positive signal => q increases by +0.4.
    m_lo.election_cost_rate = cost2
    m_hi.election_cost_rate = cost2
    m_lo.step()
    m_hi.step()

    assert float(a_lo.q_participation) == pytest.approx(0.4, abs=1e-12)
    assert float(a_hi.q_participation) == pytest.approx(0.4, abs=1e-12)

    # Step 3 decision differs by beta given the same q and the same RNG draw u3.
    m_lo.step()
    m_hi.step()

    # With low beta: p ~= sigmoid(0.2) ~ 0.55 => u3=0.7 => abstain.
    assert bool(a_lo.participating) is False
    # With high beta: p ~= sigmoid(2.0) ~ 0.88 => u3=0.7 => participate.
    assert bool(a_hi.participating) is True

