from __future__ import annotations

import numpy as np
import pytest

from tests.factory import create_test_model


pytestmark = pytest.mark.phase1


def test_participation_beta_oracle_zero_means_p_is_half_for_any_q() -> None:
    """Oracle: p = sigmoid(beta*q). With beta=0, p==0.5 for any finite q."""
    model, _ = create_test_model(seed=10, num_agents=1, num_colors=3, num_areas=1, participation_beta=0.0)
    a = model.voting_agents[0]
    assert a is not None

    for q in [-100.0, -1.0, 0.0, 1.0, 100.0]:
        a.q_participation = q
        assert float(a.participation_probability()) == pytest.approx(0.5, abs=1e-12)


def test_participation_beta_metamorphic_monotone_in_beta_for_fixed_q() -> None:
    """Metamorphic: for q>0, increasing beta increases p; for q<0, it decreases p."""
    b1 = 0.5
    b2 = 2.0
    assert b2 > b1

    m1, _ = create_test_model(seed=11, num_agents=1, num_colors=3, num_areas=1, participation_beta=b1)
    m2, _ = create_test_model(seed=11, num_agents=1, num_colors=3, num_areas=1, participation_beta=b2)
    a1 = m1.voting_agents[0]
    a2 = m2.voting_agents[0]
    assert a1 is not None and a2 is not None

    # q > 0 => higher beta => higher p
    a1.q_participation = 1.0
    a2.q_participation = 1.0
    assert float(a2.participation_probability()) > float(a1.participation_probability())

    # q < 0 => higher beta => lower p
    a1.q_participation = -1.0
    a2.q_participation = -1.0
    assert float(a2.participation_probability()) < float(a1.participation_probability())

    # q == 0 => p == 0.5 for any beta
    a1.q_participation = 0.0
    a2.q_participation = 0.0
    assert float(a1.participation_probability()) == pytest.approx(0.5, abs=1e-12)
    assert float(a2.participation_probability()) == pytest.approx(0.5, abs=1e-12)


def test_participation_beta_probability_is_well_formed() -> None:
    """Property: p is always in [0,1] and finite for finite beta,q."""
    model, _ = create_test_model(seed=12, num_agents=1, num_colors=3, num_areas=1, participation_beta=5.0)
    a = model.voting_agents[0]
    assert a is not None
    for q in np.linspace(-10, 10, 21):
        a.q_participation = float(q)
        p = float(a.participation_probability())
        assert np.isfinite(p)
        assert 0.0 <= p <= 1.0


def test_participation_beta_negative_raises() -> None:
    with pytest.raises(ValueError, match="participation_beta must be finite and >= 0"):
        create_test_model(seed=13, participation_beta=-0.01)

