from __future__ import annotations

import pytest

from tests.factory import create_test_model


pytestmark = pytest.mark.phase1


def test_altruism_init_oracle_used_only_when_learning_on() -> None:
    init_a = 0.8
    static_a = 0.2

    m_on, _ = create_test_model(
        seed=100,
        num_agents=2,
        num_colors=3,
        num_areas=1,
        altruism_learning=True,
        altruism_init=init_a,
        altruism_static=static_a,
    )
    m_off, _ = create_test_model(
        seed=100,
        num_agents=2,
        num_colors=3,
        num_areas=1,
        altruism_learning=False,
        altruism_init=init_a,
        altruism_static=static_a,
    )

    for a in m_on.voting_agents:
        if a is None:
            continue
        assert float(a.altruism_factor) == pytest.approx(init_a, abs=0.0)

    for a in m_off.voting_agents:
        if a is None:
            continue
        assert float(a.altruism_factor) == pytest.approx(static_a, abs=0.0)


def test_altruism_init_metamorphic_changes_initial_mean_when_learning_on() -> None:
    i1 = 0.1
    i2 = 0.9
    assert i2 > i1

    m1, _ = create_test_model(
        seed=101,
        num_agents=5,
        num_colors=3,
        num_areas=1,
        altruism_learning=True,
        altruism_init=i1,
    )
    m2, _ = create_test_model(
        seed=101,
        num_agents=5,
        num_colors=3,
        num_areas=1,
        altruism_learning=True,
        altruism_init=i2,
    )

    mean1 = sum(float(a.altruism_factor) for a in m1.voting_agents if a is not None) / 5.0
    mean2 = sum(float(a.altruism_factor) for a in m2.voting_agents if a is not None) / 5.0
    assert mean1 == pytest.approx(i1, abs=1e-12)
    assert mean2 == pytest.approx(i2, abs=1e-12)


def test_altruism_init_negative_or_nonfinite_raises_via_model_validation() -> None:
    with pytest.raises(ValueError, match=r"altruism_init must be finite and in \[0,1\]\."):
        create_test_model(seed=102, altruism_learning=True, altruism_init=float("nan"))
    with pytest.raises(ValueError, match=r"altruism_init must be finite and in \[0,1\]\."):
        create_test_model(seed=103, altruism_learning=True, altruism_init=-0.01)
    with pytest.raises(ValueError, match=r"altruism_init must be finite and in \[0,1\]\."):
        create_test_model(seed=104, altruism_learning=True, altruism_init=1.01)
