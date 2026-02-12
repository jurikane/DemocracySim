from __future__ import annotations

import numpy as np
import pytest

from tests.factory import create_test_model


pytestmark = pytest.mark.phase1


class _SpyNormalRng:
    """Minimal RNG stub for create_color_distribution: provides normal()."""

    def __init__(self, base: np.ndarray):
        self._base = np.asarray(base, dtype=np.float64)
        self.calls: list[tuple[float, float, int]] = []

    def normal(self, mean: float, std: float, size: int):  # type: ignore[no-untyped-def]
        self.calls.append((float(mean), float(std), int(size)))
        b = self._base
        assert b.size == int(size)
        return float(mean) + float(std) * b


def test_heterogeneity_oracle_zero_yields_uniform_distribution() -> None:
    model, _ = create_test_model(seed=600, num_colors=5, num_personality_groups=5, num_agents=5, num_areas=1, heterogeneity=0.0)
    dst = np.asarray(model.create_color_distribution(heterogeneity=0.0), dtype=np.float64)
    assert np.isclose(dst.sum(), 1.0)
    assert np.allclose(dst, np.full(5, 1.0 / 5), atol=1e-12)


def test_heterogeneity_metamorphic_increasing_heterogeneity_increases_skew_for_fixed_base_draw() -> None:
    """Metamorphic: for the same underlying standardized draw vector, larger std produces more skew."""
    model, _ = create_test_model(seed=601, num_colors=3, num_personality_groups=3, num_agents=5, num_areas=1, heterogeneity=0.0)
    base = np.asarray([-2.0, 0.0, 2.0], dtype=np.float64)
    spy = _SpyNormalRng(base)
    model.np_random = spy  # type: ignore[assignment]

    d1 = np.asarray(model.create_color_distribution(heterogeneity=0.1), dtype=np.float64)
    d2 = np.asarray(model.create_color_distribution(heterogeneity=0.5), dtype=np.float64)

    # sanity: normal() was called with mean=1.0 and std=heterogeneity
    assert spy.calls[0] == (1.0, 0.1, 3)
    assert spy.calls[1] == (1.0, 0.5, 3)

    # Skew proxy: max probability increases with heterogeneity for this base.
    assert float(d2.max()) > float(d1.max())
    assert np.isclose(d1.sum(), 1.0)
    assert np.isclose(d2.sum(), 1.0)
    assert np.all(d1 >= 0.0)
    assert np.all(d2 >= 0.0)


def test_heterogeneity_validation_negative_or_nonfinite_raises() -> None:
    with pytest.raises(ValueError, match=r"heterogeneity must be finite and >= 0"):
        create_test_model(seed=602, heterogeneity=-0.01)
    with pytest.raises(ValueError, match=r"heterogeneity must be finite and >= 0"):
        create_test_model(seed=603, heterogeneity=float("nan"))

