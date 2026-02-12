from __future__ import annotations

import numpy as np
import pytest

from tests.factory import create_test_model


pytestmark = pytest.mark.phase1


def _model(**overrides):
    base = dict(
        seed=110,
        num_agents=1,
        num_colors=3,
        num_areas=1,
        altruism_learning=True,
        altruism_alpha=1.0,
        altruism_init=0.5,
        altruism_clip_min=0.0,
        altruism_clip_max=1.0,
    )
    base.update(overrides)
    model, _ = create_test_model(**base)
    return model


def test_altruism_clip_oracle_clips_at_bounds() -> None:
    model = _model(altruism_clip_min=0.2, altruism_clip_max=0.8, altruism_alpha=1.0)
    a = model.voting_agents[0]
    assert a is not None
    a._participating = True

    # Push above max.
    a.altruism_factor = 0.7
    a.apply_altruism_update(satisfaction_signal=+1.0)  # would go to 1.7
    assert float(a.altruism_factor) == pytest.approx(0.8, abs=0.0)

    # Push below min.
    a.altruism_factor = 0.3
    a.apply_altruism_update(satisfaction_signal=-1.0)  # would go to -0.7
    assert float(a.altruism_factor) == pytest.approx(0.2, abs=0.0)


def test_altruism_clip_metamorphic_tightening_bounds_restricts_outcome() -> None:
    """Metamorphic: narrowing [min,max] cannot produce altruism outside the narrower interval."""
    sig = +1.0
    alpha = 1.0
    init = 0.5

    m_wide = _model(seed=111, altruism_init=init, altruism_alpha=alpha, altruism_clip_min=0.0, altruism_clip_max=1.0)
    m_narrow = _model(seed=111, altruism_init=init, altruism_alpha=alpha, altruism_clip_min=0.4, altruism_clip_max=0.6)
    a_w = m_wide.voting_agents[0]
    a_n = m_narrow.voting_agents[0]
    assert a_w is not None and a_n is not None
    a_w._participating = True
    a_n._participating = True

    a_w.apply_altruism_update(satisfaction_signal=sig)  # would go to 1.5 => clipped to 1.0
    a_n.apply_altruism_update(satisfaction_signal=sig)  # would go to 1.5 => clipped to 0.6

    assert 0.0 <= float(a_w.altruism_factor) <= 1.0
    assert 0.4 <= float(a_n.altruism_factor) <= 0.6
    assert float(a_n.altruism_factor) <= float(a_w.altruism_factor) + 1e-12


def test_altruism_clip_validation_min_le_max_and_finite() -> None:
    with pytest.raises(ValueError, match=r"altruism_clip_min/max must be finite and satisfy clip_min <= clip_max"):
        _model(altruism_clip_min=0.9, altruism_clip_max=0.1)
    with pytest.raises(ValueError, match=r"altruism_clip_min/max must be finite and satisfy clip_min <= clip_max"):
        _model(altruism_clip_min=float("nan"), altruism_clip_max=1.0)
    with pytest.raises(ValueError, match=r"altruism_clip_min/max must be finite and satisfy clip_min <= clip_max"):
        _model(altruism_clip_min=0.0, altruism_clip_max=float("inf"))
