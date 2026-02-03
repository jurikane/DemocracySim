import numpy as np
import random

from tests.factory import create_test_model


def _sigmoid(x: float) -> float:
    # Local helper to avoid importing private helper.
    return float(1.0 / (1.0 + np.exp(-x)))


def test_q_and_probability_are_well_formed_after_steps():
    model, _ = create_test_model(
        seed=123,
        participation_alpha=0.2,
        participation_beta=2.0,
        participation_init_q=0.0,
        participation_q_max=5.0,
        max_steps=10,
    )

    # Step a few times; learning should update q after elections.
    for _ in range(3):
        model.step()

    for a in model.voting_agents:
        assert np.isfinite(float(a.q_participation))
        p = a.participation_probability()
        assert 0.0 <= p <= 1.0
        # Ensure clipping bound is respected.
        assert abs(float(a.q_participation)) <= 5.0 + 1e-9


def test_learning_direction_positive_delta_increases_probability():
    # Build a tiny model; we only need model parameters + RNG.
    model, _ = create_test_model(
        seed=1,
        participation_alpha=0.5,
        participation_beta=1.0,
        participation_init_q=0.0,
        participation_q_max=100.0,
    )

    agent = model.voting_agents[0]
    q0 = float(agent.q_participation)

    # Participated: positive delta should increase q/p.
    agent.q_participation = q0
    agent._participating = True
    agent.apply_participation_update(delta_assets=+10.0)
    q_pos = float(agent.q_participation)
    p_pos = agent.participation_probability()

    # Abstained: same positive delta should decrease q/p.
    agent.q_participation = q0
    agent._participating = False
    agent.apply_participation_update(delta_assets=+10.0)
    q_abs = float(agent.q_participation)
    p_abs = agent.participation_probability()

    assert q_pos > q0
    assert p_pos > 0.5
    assert q_abs < q0
    assert p_abs < 0.5


def test_determinism_same_seed_produces_same_participation_counts():
    # We use num_elections_participated as a proxy for participation decisions.
    # Note: model init still touches global RNG state in a few legacy spots,
    # so reset those streams between constructions.
    kwargs = dict(
        seed=999,
        participation_alpha=0.2,
        participation_beta=1.5,
        participation_init_q=0.0,
        participation_q_max=5.0,
        max_steps=10,
    )

    np.random.seed(kwargs["seed"])
    random.seed(kwargs["seed"])
    m1, _ = create_test_model(**kwargs)

    np.random.seed(kwargs["seed"])
    random.seed(kwargs["seed"])
    m2, _ = create_test_model(**kwargs)

    for _ in range(3):
        m1.step()
        m2.step()

    counts1 = [int(a.num_elections_participated) for a in m1.voting_agents]
    counts2 = [int(a.num_elections_participated) for a in m2.voting_agents]

    assert counts1 == counts2
