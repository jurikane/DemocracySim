from __future__ import annotations

import numpy as np
import pytest

from tests.factory import create_test_model


pytestmark = pytest.mark.phase1


class _AlwaysParticipate:
    def decide_participation(self, agent, area) -> bool:  # type: ignore[no-untyped-def]
        return True


class _ZeroBallot:
    def score_options(self, agent, area, options) -> np.ndarray:  # type: ignore[no-untyped-def]
        return np.zeros(int(options.shape[0]), dtype=np.float32)


class _SpyNpRandom:
    """Wrap a NumPy Generator-like object and intercept choice()."""

    def __init__(self, base, *, choice_impl):
        self._base = base
        self._choice_impl = choice_impl

    def choice(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return self._choice_impl(*args, **kwargs)

    def random(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return self._base.random(*args, **kwargs)


def _model_one_area(*, impact: float, **overrides):
    base = dict(
        seed=400,
        num_colors=3,
        num_personality_groups=3,
        num_agents=6,
        num_areas=1,
        height=8,
        width=8,
        av_area_height=8,
        av_area_width=8,
        area_size_variance=0.0,
        known_cells=0,
        max_steps=3,
        mu=1.0,
        election_impact_on_mutation=impact,
        election_cost_rate=0.0,
        reward_rate_common=0.0,
        reward_rate_personal=0.0,
    )
    base.update(overrides)
    model, _ = create_test_model(**base)
    for a in model.voting_agents:
        if a is None:
            continue
        a.participation_strategy = _AlwaysParticipate()
        a.voting_strategy = _ZeroBallot()
    return model


def test_election_impact_oracle_zero_means_uniform_color_probs() -> None:
    c = 5
    model, _ = create_test_model(
        seed=401,
        num_colors=c,
        num_personality_groups=c,
        num_agents=5,
        num_areas=1,
        election_impact_on_mutation=0.0,
        mu=0.0,
        known_cells=0,
        height=8,
        width=8,
        av_area_height=8,
        av_area_width=8,
        area_size_variance=0.0,
        election_cost_rate=0.0,
    )
    p = np.asarray(model.color_probs, dtype=np.float64)
    assert p.shape == (c,)
    assert np.allclose(p, np.full(c, 1.0 / c), atol=1e-12)


def test_election_impact_metamorphic_increasing_impact_increases_top_mass() -> None:
    m1 = _model_one_area(impact=0.5)
    m2 = _model_one_area(impact=2.0)
    p1 = np.asarray(m1.color_probs, dtype=np.float64)
    p2 = np.asarray(m2.color_probs, dtype=np.float64)

    # More impact => more skew towards earlier ranks in voted_ordering.
    assert p2[0] > p1[0]
    assert p2[-1] < p1[-1]
    assert np.isclose(p1.sum(), 1.0)
    assert np.isclose(p2.sum(), 1.0)


def test_election_impact_integration_mutate_cells_passes_color_probs_to_choice() -> None:
    """Integration: mutate_cells must call np_random.choice(voted_ordering, p=color_probs)."""
    model = _model_one_area(impact=1.7)
    area = model.areas[0]

    # Step 1: produce a valid election and enable mutation on step 2.
    model.step()
    # Force a known voted_ordering so we can assert the argument passed to choice.
    area._voted_ordering = np.asarray([2, 1, 0], dtype=np.int64)
    area._voter_turnout = 100

    expected_p = np.asarray(model.color_probs, dtype=np.float64)
    expected_a = np.asarray(area.voted_ordering, dtype=np.int64)

    calls = {"seen": False}

    def _choice(a, *, size, p):  # type: ignore[no-untyped-def]
        aa = np.asarray(a, dtype=np.int64)
        pp = np.asarray(p, dtype=np.float64)
        assert np.array_equal(aa, expected_a)
        assert np.allclose(pp, expected_p)
        calls["seen"] = True
        # Deterministic: always pick the first element.
        return np.full(int(size), int(aa[0]), dtype=np.int64)

    # Patch only for step 2.
    model.np_random = _SpyNpRandom(model.np_random, choice_impl=_choice)

    # Ensure mutations are visible: set all cells to 0 first, then expect all become 2.
    for c in area.cells:
        c.color = 0
    area.update_color_distribution()

    model.step()  # step 2 triggers mutate_cells at scheduler start
    assert calls["seen"] is True
    assert all(int(c.color) == 2 for c in area.cells)


def test_election_impact_negative_or_nonfinite_raises() -> None:
    with pytest.raises(ValueError, match=r"election_impact_on_mutation must be finite and >= 0"):
        _model_one_area(impact=-0.1)
    with pytest.raises(ValueError, match=r"election_impact_on_mutation must be finite and >= 0"):
        _model_one_area(impact=float("nan"))
