import pytest

from tests.factory import create_test_model


def test_color_patches_steps_validation_requires_int_ge_0():
    with pytest.raises(ValueError):
        create_test_model(color_patches_steps=-1)
    with pytest.raises(ValueError):
        create_test_model(color_patches_steps=1.5)
    with pytest.raises(ValueError):
        create_test_model(color_patches_steps=True)


def test_color_patches_steps_stored_on_model():
    model, _ = create_test_model(height=5, width=5, num_agents=2, num_areas=1,
                                 av_area_height=5, av_area_width=5,
                                 area_size_variance=0.0,
                                 num_colors=3, num_personality_groups=1,
                                 known_cells=1,
                                 color_patches_steps=0,
                                 seed=123)
    assert model.color_patches_steps == 0


def test_adjust_color_pattern_steps_0_is_noop():
    model, _ = create_test_model(height=5, width=5, num_agents=2, num_areas=1,
                                 av_area_height=5, av_area_width=5,
                                 area_size_variance=0.0,
                                 num_colors=3, num_personality_groups=1,
                                 known_cells=1,
                                 color_patches_steps=0,
                                 seed=123)
    before = [c.color for c in model.color_cells]
    model.adjust_color_pattern(0, patch_power=1.0)
    after = [c.color for c in model.color_cells]
    assert after == before


def test_adjust_color_pattern_calls_color_patches_once_per_cell_per_step(monkeypatch):
    model, _ = create_test_model(height=4, width=4, num_agents=2, num_areas=1,
                                 av_area_height=4, av_area_width=4,
                                 area_size_variance=0.0,
                                 num_colors=3, num_personality_groups=1,
                                 known_cells=1,
                                 color_patches_steps=0,
                                 seed=123)

    calls = {"n": 0}

    def _color_patches_no_change(cell, patch_power):
        calls["n"] += 1
        return cell.color

    # Make the loop deterministic (order doesn't matter for call-count).
    monkeypatch.setattr(model.random, "shuffle", lambda xs: None)
    monkeypatch.setattr(model, "color_patches", _color_patches_no_change)

    steps = 3
    model.adjust_color_pattern(steps, patch_power=1.0)
    assert calls["n"] == steps * (model.height * model.width)

