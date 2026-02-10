from __future__ import annotations

import numpy as np

from tests.factory import create_test_model


def test_action_reinforcement_positive_delta_participant_up_abstainer_down() -> None:
    """Contract: naive action reinforcement.

    With participation_signal > 0:
    - participating agent should increase q => p increases
    - abstained agent should decrease q => p decreases

    This test is unit-level on the update rule; it does not require a full election.
    """

    model, _ = create_test_model(seed=123, num_agents=2, num_colors=3, num_areas=1)

    a0 = model.voting_agents[0]
    a1 = model.voting_agents[1]
    assert a0 is not None and a1 is not None

    # Start at identical state
    a0.q_participation = 0.0
    a1.q_participation = 0.0
    model.participation_alpha = 0.5
    model.participation_beta = 1.0

    p0_before = a0.participation_probability()
    p1_before = a1.participation_probability()
    assert np.isclose(p0_before, 0.5)
    assert np.isclose(p1_before, 0.5)

    # Mark actions
    a0._participating = True   # participant
    a1._participating = False  # abstainer

    delta = 2.0
    # Learning now uses relative delta; magnitude doesn't matter for monotonicity.
    a0.apply_participation_update(participation_signal=delta)
    a1.apply_participation_update(participation_signal=delta)

    p0_after = a0.participation_probability()
    p1_after = a1.participation_probability()

    assert p0_after > p0_before
    assert p1_after < p1_before


def test_action_reinforcement_negative_delta_participant_down_abstainer_up() -> None:
    """Contract: naive action reinforcement with negative outcomes."""

    model, _ = create_test_model(seed=123, num_agents=2, num_colors=3, num_areas=1)

    a0 = model.voting_agents[0]
    a1 = model.voting_agents[1]
    assert a0 is not None and a1 is not None

    a0.q_participation = 0.0
    a1.q_participation = 0.0
    model.participation_alpha = 0.5
    model.participation_beta = 1.0

    p0_before = a0.participation_probability()
    p1_before = a1.participation_probability()

    a0._participating = True
    a1._participating = False

    delta = -2.0
    a0.apply_participation_update(participation_signal=delta)
    a1.apply_participation_update(participation_signal=delta)

    p0_after = a0.participation_probability()
    p1_after = a1.participation_probability()

    assert p0_after < p0_before
    assert p1_after > p1_before
