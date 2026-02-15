from __future__ import annotations

import pytest

from tests.factory import create_test_model


def test_satisfaction_mode_allows_combination() -> None:
    model, _ = create_test_model(satisfaction_mode="combination")
    assert model.satisfaction_mode == "combination"


def test_satisfaction_baseline_alpha_bounds() -> None:
    with pytest.raises(ValueError):
        create_test_model(satisfaction_baseline_alpha=-0.1)
    with pytest.raises(ValueError):
        create_test_model(satisfaction_baseline_alpha=1.1)


def test_participation_baseline_alpha_bounds() -> None:
    with pytest.raises(ValueError):
        create_test_model(participation_baseline_alpha=-0.1)
    with pytest.raises(ValueError):
        create_test_model(participation_baseline_alpha=1.1)


def test_dissatisfaction_values_are_initialized() -> None:
    model, _ = create_test_model()
    agent = model.voting_agents[0]
    assert isinstance(agent.dissatisfaction_value, float)
    assert isinstance(agent.dissatisfaction_baseline, float)
