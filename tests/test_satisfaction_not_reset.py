from __future__ import annotations

from tests.factory import create_test_model


def test_reset_reward_variables_does_not_reset_satisfaction() -> None:
    model, _ = create_test_model(num_agents=1, num_areas=1)
    agent = model.voting_agents[0]

    agent.satisfaction_value = 0.42
    agent.satisfaction_baseline = 0.31
    agent.satisfaction_signal = 0.11
    agent.participation_baseline = 0.25
    agent.participation_signal = 0.05

    agent.reset_reward_variables()

    assert agent.satisfaction_value == 0.42
    assert agent.satisfaction_baseline == 0.31
    assert agent.satisfaction_signal == 0.11
    assert agent.participation_baseline == 0.25
    assert agent.participation_signal == 0.05
