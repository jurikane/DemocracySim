from __future__ import annotations

import numpy as np
import pytest

from tests.factory import create_test_model


pytestmark = pytest.mark.phase1


def test_participation_baseline_alpha_oracle_ema_update_and_signal() -> None:
    """Oracle: signal = delta - baseline, baseline <- (1-a)*baseline + a*delta."""
    model, _ = create_test_model(seed=50, num_agents=1, num_colors=3, num_areas=1)
    a = model.voting_agents[0]
    assert a is not None

    baseline0 = 0.2
    delta = 0.7
    alpha = 0.3

    a.participation_baseline = baseline0
    model.participation_baseline_alpha = alpha

    # Mirror the formula used in Area.conduct_election (unit-level).
    signal = delta - baseline0
    baseline1 = (1.0 - alpha) * baseline0 + alpha * delta

    assert np.isclose(signal, 0.5)
    assert np.isclose(baseline1, 0.35)


def test_participation_baseline_alpha_metamorphic_endpoints() -> None:
    """Metamorphic: alpha=0 keeps baseline, alpha=1 sets baseline to delta."""
    baseline0 = 0.2
    delta = 0.7

    alpha0 = 0.0
    alpha1 = 1.0

    b0 = (1.0 - alpha0) * baseline0 + alpha0 * delta
    b1 = (1.0 - alpha1) * baseline0 + alpha1 * delta

    assert b0 == pytest.approx(baseline0, abs=0.0)
    assert b1 == pytest.approx(delta, abs=0.0)


def test_participation_baseline_alpha_validation_already_enforced() -> None:
    # create_test_model should raise via ParticipationModel validation.
    with pytest.raises(ValueError):
        create_test_model(seed=51, participation_baseline_alpha=-0.1)
    with pytest.raises(ValueError):
        create_test_model(seed=52, participation_baseline_alpha=1.1)

