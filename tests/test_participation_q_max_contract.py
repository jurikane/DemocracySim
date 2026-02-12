from __future__ import annotations

import pytest

from tests.factory import create_test_model


pytestmark = pytest.mark.phase1


def test_participation_q_max_oracle_clips_q_symmetrically() -> None:
    """Oracle: with q_max>0, q is clipped to [-q_max, q_max] after an update."""
    q_max = 1.0
    model, _ = create_test_model(
        seed=30,
        num_agents=2,
        num_colors=3,
        num_areas=1,
        participation_alpha=1.0,
        participation_q_max=q_max,
    )
    a_part = model.voting_agents[0]
    a_abs = model.voting_agents[1]
    assert a_part is not None and a_abs is not None

    a_part.q_participation = 0.0
    a_abs.q_participation = 0.0
    a_part._participating = True
    a_abs._participating = False

    # Huge signal => unclipped would exceed bounds.
    a_part.apply_participation_update(participation_signal=10.0)
    a_abs.apply_participation_update(participation_signal=10.0)

    assert float(a_part.q_participation) == pytest.approx(+q_max, abs=0.0)
    assert float(a_abs.q_participation) == pytest.approx(-q_max, abs=0.0)


def test_participation_q_max_metamorphic_smaller_cap_implies_smaller_abs_q() -> None:
    """Metamorphic: lowering q_max cannot increase the resulting |q| (same alpha, signal, action)."""
    alpha = 1.0
    signal = 2.5
    q_max_small = 0.5
    q_max_large = 5.0
    assert q_max_large > q_max_small

    m_small, _ = create_test_model(
        seed=31,
        num_agents=1,
        num_colors=3,
        num_areas=1,
        participation_alpha=alpha,
        participation_q_max=q_max_small,
    )
    m_large, _ = create_test_model(
        seed=31,
        num_agents=1,
        num_colors=3,
        num_areas=1,
        participation_alpha=alpha,
        participation_q_max=q_max_large,
    )
    a_s = m_small.voting_agents[0]
    a_l = m_large.voting_agents[0]
    assert a_s is not None and a_l is not None

    a_s.q_participation = 0.0
    a_l.q_participation = 0.0
    a_s._participating = True
    a_l._participating = True

    a_s.apply_participation_update(participation_signal=signal)
    a_l.apply_participation_update(participation_signal=signal)

    assert abs(float(a_s.q_participation)) <= abs(float(a_l.q_participation)) + 1e-12
    assert abs(float(a_s.q_participation)) <= q_max_small + 1e-12
    assert abs(float(a_l.q_participation)) <= q_max_large + 1e-12


def test_participation_q_max_zero_disables_clipping() -> None:
    """Documented behavior: if q_max == 0, no clipping is applied (q can move freely)."""
    model, _ = create_test_model(
        seed=32,
        num_agents=1,
        num_colors=3,
        num_areas=1,
        participation_alpha=1.0,
        participation_q_max=0.0,
    )
    a = model.voting_agents[0]
    assert a is not None
    a.q_participation = 0.0
    a._participating = True
    a.apply_participation_update(participation_signal=10.0)
    assert float(a.q_participation) == pytest.approx(10.0, abs=1e-12)


def test_participation_q_max_invalid_raises() -> None:
    with pytest.raises(ValueError, match="participation_q_max must be finite and >= 0"):
        create_test_model(seed=33, participation_q_max=-0.1)
    with pytest.raises(ValueError, match="participation_q_max must be finite and >= 0"):
        create_test_model(seed=34, participation_q_max=float("nan"))

