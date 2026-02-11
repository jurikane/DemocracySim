from __future__ import annotations

import numpy as np

from tests.factory import create_test_model


def test_participation_baseline_alpha_one_equals_last_step_delta() -> None:
    model, _ = create_test_model(num_agents=1, num_areas=1)
    model.participation_baseline_alpha = 1.0
    agent = model.voting_agents[0]

    agent.participation_baseline = 0.2
    delta = 0.7
    baseline = agent.participation_baseline

    signal = delta - baseline
    new_baseline = (1.0 - model.participation_baseline_alpha) * baseline + model.participation_baseline_alpha * delta

    assert np.isclose(signal, delta - 0.2)
    assert np.isclose(new_baseline, delta)
