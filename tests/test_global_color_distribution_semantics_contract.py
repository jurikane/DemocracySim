import numpy as np

from tests.factory import create_test_model


def _grid_distribution_from_cells(model) -> np.ndarray:
    counts = np.zeros(model.num_colors, dtype=np.float64)
    for c in model.color_cells:
        counts[int(c.color)] += 1.0
    return counts / float(len(model.color_cells))


def test_global_color_dst_matches_grid_after_init():
    # Important for step-0 / initialization logs: global_color_dst must reflect the realized grid.
    model, _ = create_test_model(height=6, width=6, num_agents=2, num_areas=1,
                                 av_area_height=6, av_area_width=6,
                                 area_size_variance=0.0,
                                 num_colors=3, num_personality_groups=1,
                                 known_cells=1,
                                 color_patches_steps=3, patch_power=1.0,
                                 seed=123)
    expected = _grid_distribution_from_cells(model)
    np.testing.assert_allclose(model.global_color_dst, expected, rtol=0, atol=0)
    np.testing.assert_allclose(model.global_color_dst.sum(), 1.0, rtol=0, atol=1e-12)


def test_update_global_color_distribution_is_exact_for_disjoint_areas_with_gaps():
    # Areas may be disjoint but not cover the full grid (gaps). We still require
    # global_color_dst to match the realized grid distribution exactly.
    model, _ = create_test_model(height=6, width=6, num_agents=2, num_areas=1,
                                 av_area_height=2, av_area_width=2,  # area covers only part of grid
                                 area_size_variance=0.0,
                                 num_colors=3, num_personality_groups=1,
                                 known_cells=1,
                                 color_patches_steps=0,
                                 seed=123)
    model.update_global_color_distribution()
    expected = _grid_distribution_from_cells(model)
    np.testing.assert_allclose(model.global_color_dst, expected, rtol=0, atol=0)


def test_update_global_color_distribution_falls_back_to_grid_when_marked_not_disjoint():
    # If areas overlap, cached area-count shortcuts are unsafe; ensure grid counting is used.
    model, _ = create_test_model(height=5, width=5, num_agents=2, num_areas=1,
                                 av_area_height=5, av_area_width=5,
                                 area_size_variance=0.0,
                                 num_colors=3, num_personality_groups=1,
                                 known_cells=1,
                                 color_patches_steps=0,
                                 seed=123)
    expected = _grid_distribution_from_cells(model)
    # Poison the cached shortcut inputs and force the overlap path.
    model._areas_are_disjoint = False
    model._uncovered_color_counts = np.array([999, 999, 999], dtype=np.int64)
    for a in model.areas:
        if a.unique_id != -1:
            a._color_counts = np.array([999, 999, 999], dtype=np.int64)
    model.update_global_color_distribution()
    np.testing.assert_allclose(model.global_color_dst, expected, rtol=0, atol=0)
