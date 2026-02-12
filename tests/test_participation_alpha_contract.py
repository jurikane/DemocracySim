from __future__ import annotations

import pytest

from tests.factory import create_test_model


pytestmark = pytest.mark.phase1


def test_participation_alpha_oracle_exact_q_update_participant_and_abstainer() -> None:
    """Oracle: q update is exactly q += alpha * sign * signal (with sign based on last action)."""
    alpha = 0.25
    signal = 2.0
    model, _ = create_test_model(
        seed=1,
        num_agents=2,
        num_colors=3,
        num_areas=1,
        participation_alpha=alpha,
        participation_q_max=1_000.0,
    )
    a_part = model.voting_agents[0]
    a_abs = model.voting_agents[1]
    assert a_part is not None and a_abs is not None

    a_part.q_participation = 1.0
    a_abs.q_participation = 1.0

    a_part._participating = True
    a_abs._participating = False

    a_part.apply_participation_update(participation_signal=signal)
    a_abs.apply_participation_update(participation_signal=signal)

    assert float(a_part.q_participation) == pytest.approx(1.0 + alpha * signal, abs=1e-12)
    assert float(a_abs.q_participation) == pytest.approx(1.0 - alpha * signal, abs=1e-12)


def test_participation_alpha_metamorphic_ratio_two_runs() -> None:
    """Metamorphic: scaling participation_alpha scales delta-q proportionally (away from clipping)."""
    alpha1 = 0.1
    alpha2 = 0.4
    assert alpha2 > alpha1
    ratio = alpha2 / alpha1
    signal = 3.0

    m1, _ = create_test_model(seed=2, num_agents=1, num_colors=3, num_areas=1, participation_alpha=alpha1, participation_q_max=1_000.0)
    m2, _ = create_test_model(seed=2, num_agents=1, num_colors=3, num_areas=1, participation_alpha=alpha2, participation_q_max=1_000.0)
    a1 = m1.voting_agents[0]
    a2 = m2.voting_agents[0]
    assert a1 is not None and a2 is not None

    a1.q_participation = 0.0
    a2.q_participation = 0.0
    a1._participating = True
    a2._participating = True

    a1.apply_participation_update(participation_signal=signal)
    a2.apply_participation_update(participation_signal=signal)

    dq1 = float(a1.q_participation) - 0.0
    dq2 = float(a2.q_participation) - 0.0
    assert (dq2 / dq1) == pytest.approx(ratio, rel=1e-12, abs=1e-12)


def test_participation_alpha_zero_means_no_update() -> None:
    model, _ = create_test_model(seed=3, num_agents=1, num_colors=3, num_areas=1, participation_alpha=0.0, participation_q_max=1_000.0)
    a = model.voting_agents[0]
    assert a is not None

    a.q_participation = 4.0
    a._participating = True
    a.apply_participation_update(participation_signal=999.0)
    assert float(a.q_participation) == pytest.approx(4.0, abs=0.0)


def test_participation_alpha_negative_raises() -> None:
    with pytest.raises(ValueError, match=r"Learning rates alpha must be finite and >= 0\."):
        create_test_model(seed=4, participation_alpha=-0.01)
