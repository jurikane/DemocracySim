from __future__ import annotations

import numpy as np
import pytest

from tests.factory import create_test_model


pytestmark = pytest.mark.phase1


def test_satisfaction_baseline_alpha_oracle_signal_and_ema_update() -> None:
    """Oracle: signal = sv - baseline, baseline <- (1-a)*baseline + a*sv."""
    model, _ = create_test_model(
        seed=130,
        num_agents=1,
        num_colors=2,
        num_personality_groups=2,
        num_areas=1,
    )
    a = model.voting_agents[0]
    assert a is not None

    baseline0 = 0.2
    sv = 0.7
    alpha = 0.3
    model.satisfaction_baseline_alpha = alpha

    a.satisfaction_baseline = baseline0
    signal = sv - baseline0
    baseline1 = (1.0 - alpha) * baseline0 + alpha * sv

    assert np.isclose(signal, 0.5)
    assert np.isclose(baseline1, 0.35)


def test_satisfaction_baseline_alpha_metamorphic_endpoints() -> None:
    baseline0 = 0.2
    sv = 0.7
    alpha0 = 0.0
    alpha1 = 1.0

    b0 = (1.0 - alpha0) * baseline0 + alpha0 * sv
    b1 = (1.0 - alpha1) * baseline0 + alpha1 * sv

    assert b0 == pytest.approx(baseline0, abs=0.0)
    assert b1 == pytest.approx(sv, abs=0.0)
