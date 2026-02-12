from __future__ import annotations

import numpy as np
import pytest

from tests.factory import create_test_model


pytestmark = pytest.mark.phase1


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def test_participation_init_q_oracle_sets_initial_q_and_probability() -> None:
    init_q = 1.25
    beta = 2.0
    model, _ = create_test_model(
        seed=20,
        num_agents=1,
        num_colors=3,
        num_areas=1,
        participation_init_q=init_q,
        participation_beta=beta,
    )
    a = model.voting_agents[0]
    assert a is not None

    assert float(a.q_participation) == pytest.approx(init_q, abs=0.0)
    assert float(a.participation_probability()) == pytest.approx(_sigmoid(beta * init_q), abs=1e-12)


def test_participation_init_q_metamorphic_increases_initial_p_for_beta_positive() -> None:
    beta = 1.5
    q1 = -0.5
    q2 = 0.5
    assert q2 > q1

    m1, _ = create_test_model(seed=21, num_agents=1, num_colors=3, num_areas=1, participation_init_q=q1, participation_beta=beta)
    m2, _ = create_test_model(seed=21, num_agents=1, num_colors=3, num_areas=1, participation_init_q=q2, participation_beta=beta)
    a1 = m1.voting_agents[0]
    a2 = m2.voting_agents[0]
    assert a1 is not None and a2 is not None

    assert float(a2.participation_probability()) > float(a1.participation_probability())


def test_participation_init_q_nonfinite_raises() -> None:
    with pytest.raises(ValueError, match="participation_init_q must be finite"):
        create_test_model(seed=22, participation_init_q=float("nan"))
    with pytest.raises(ValueError, match="participation_init_q must be finite"):
        create_test_model(seed=23, participation_init_q=float("inf"))

