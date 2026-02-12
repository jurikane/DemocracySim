import math

import pytest

from tests.factory import create_test_model


def test_patch_power_validation_requires_finite_ge_0():
    with pytest.raises(ValueError):
        create_test_model(patch_power=-0.1)
    with pytest.raises(ValueError):
        create_test_model(patch_power=float("nan"))
    with pytest.raises(ValueError):
        create_test_model(patch_power=float("inf"))


def test_patch_power_stored_on_model():
    model, _ = create_test_model(height=5, width=5, num_agents=2, num_areas=1,
                                 av_area_height=5, av_area_width=5,
                                 area_size_variance=0.0,
                                 num_colors=3, num_personality_groups=1,
                                 known_cells=1,
                                 color_patches_steps=0,
                                 patch_power=1.25,
                                 seed=123)
    assert math.isclose(model.patch_power, 1.25)


def test_color_patches_uses_patch_power_as_gauss_sigma():
    model, _ = create_test_model(height=5, width=5, num_agents=2, num_areas=1,
                                 av_area_height=5, av_area_width=5,
                                 area_size_variance=0.0,
                                 num_colors=3, num_personality_groups=1,
                                 known_cells=1,
                                 color_patches_steps=0,
                                 seed=123)

    # Pick a cell with guaranteed bias_factor > 0 by forcing biases to 0 and
    # selecting a non-(0,0) cell.
    model._horizontal_bias = 0.0
    model._vertical_bias = 0.0
    cell = next(c for c in model.color_cells if (c.row != 0 or c.col != 0))

    seen_sigmas: list[float] = []

    def _gauss(mu, sigma):
        seen_sigmas.append(float(sigma))
        return 0.0

    model.random.gauss = _gauss  # type: ignore[assignment]

    sigma = 0.7
    _ = model.color_patches(cell, patch_power=sigma)
    assert seen_sigmas == [sigma]


def test_patch_power_zero_forces_preset_distribution_path_for_non_bias_cell(monkeypatch):
    model, _ = create_test_model(height=5, width=5, num_agents=2, num_areas=1,
                                 av_area_height=5, av_area_width=5,
                                 area_size_variance=0.0,
                                 num_colors=3, num_personality_groups=1,
                                 known_cells=1,
                                 color_patches_steps=0,
                                 seed=123)

    # Ensure bias_factor > 0 for chosen cell.
    model._horizontal_bias = 0.0
    model._vertical_bias = 0.0
    cell = next(c for c in model.color_cells if (c.row != 0 or c.col != 0))

    # If neighbor path is taken, this should fail loudly.
    monkeypatch.setattr(model.grid, "get_neighbors", lambda *a, **k: (_ for _ in ()).throw(AssertionError("neighbor path should not be used when patch_power=0 for bias_factor>0")))

    sentinel = 2
    monkeypatch.setattr(model, "color_by_dst_rng", lambda *a, **k: sentinel)

    assert model.color_patches(cell, patch_power=0.0) == sentinel

