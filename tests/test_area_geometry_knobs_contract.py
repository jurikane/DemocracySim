import numpy as np
import pytest

from tests.factory import create_test_model

def test_av_area_height_width_validation_fail_loud():
    with pytest.raises(ValueError):
        create_test_model(av_area_height=0)
    with pytest.raises(ValueError):
        create_test_model(av_area_width=0)
    with pytest.raises(ValueError):
        create_test_model(av_area_height=-1)
    with pytest.raises(ValueError):
        create_test_model(av_area_width=-1)
    with pytest.raises(ValueError):
        create_test_model(av_area_height=1.5)
    with pytest.raises(ValueError):
        create_test_model(av_area_width=1.5)
    with pytest.raises(ValueError):
        create_test_model(av_area_height=True)
    with pytest.raises(ValueError):
        create_test_model(av_area_width=True)


def test_av_area_height_width_validation_bound_by_grid():
    with pytest.raises(ValueError):
        create_test_model(height=8, width=8, av_area_height=9)
    with pytest.raises(ValueError):
        create_test_model(height=8, width=8, av_area_width=9)


def test_area_size_variance_validation_fail_loud():
    with pytest.raises(ValueError):
        create_test_model(area_size_variance=-0.1)
    with pytest.raises(ValueError):
        create_test_model(area_size_variance=1.1)
    with pytest.raises(ValueError):
        create_test_model(area_size_variance=float("nan"))
    with pytest.raises(ValueError):
        create_test_model(area_size_variance=float("inf"))


def test_av_area_height_metamorphic_changes_area_cell_count_when_variance_zero():
    m_small, _ = create_test_model(
        seed=801, height=12, width=12, num_areas=4,
        av_area_height=3, av_area_width=4, area_size_variance=0.0
    )
    m_large, _ = create_test_model(
        seed=801, height=12, width=12, num_areas=4,
        av_area_height=6, av_area_width=4, area_size_variance=0.0
    )
    cells_small = [a.num_cells for a in m_small.areas]
    cells_large = [a.num_cells for a in m_large.areas]
    assert cells_small and cells_large
    assert all(c == cells_small[0] for c in cells_small)
    assert all(c == cells_large[0] for c in cells_large)
    assert cells_large[0] == 2 * cells_small[0]


def test_av_area_width_metamorphic_changes_area_cell_count_when_variance_zero():
    m_small, _ = create_test_model(
        seed=802, height=12, width=12, num_areas=4,
        av_area_height=4, av_area_width=3, area_size_variance=0.0
    )
    m_large, _ = create_test_model(
        seed=802, height=12, width=12, num_areas=4,
        av_area_height=4, av_area_width=6, area_size_variance=0.0
    )
    cells_small = [a.num_cells for a in m_small.areas]
    cells_large = [a.num_cells for a in m_large.areas]
    assert cells_small and cells_large
    assert all(c == cells_small[0] for c in cells_small)
    assert all(c == cells_large[0] for c in cells_large)
    assert cells_large[0] == 2 * cells_small[0]


def test_area_size_variance_metamorphic_no_overlap_flag_behavior():
    base = dict(
        seed=803, height=12, width=12, num_areas=4,
        av_area_height=6, av_area_width=6
    )
    m0, _ = create_test_model(**base, area_size_variance=0.0)
    m1, _ = create_test_model(**base, area_size_variance=0.2)
    assert m0.no_overlap is True
    mem0 = np.zeros((m0.height, m0.width), dtype=np.int32)
    for a in m0.areas:
        for c in a.cells:
            mem0[int(c.row), int(c.col)] += 1
    assert np.all(mem0 == 1), "Variance 0 with this setup must form an exact partition."

    mem1 = np.zeros((m1.height, m1.width), dtype=np.int32)
    for a in m1.areas:
        for c in a.cells:
            mem1[int(c.row), int(c.col)] += 1
    assert np.any(mem1 != 1), "With variance>0 this setup must no longer be an exact partition."


def test_area_size_variance_integration_no_zero_sized_areas_and_valid_distributions():
    model, _ = create_test_model(
        seed=804, height=20, width=20, num_areas=16,
        av_area_height=5, av_area_width=5, area_size_variance=1.0,
        num_agents=40
    )
    # All areas must remain valid even at max variance.
    for area in model.areas:
        assert area.num_cells >= 1
        assert area._width >= 1
        assert area._height >= 1
        assert np.isfinite(area.color_distribution).all()
        np.testing.assert_allclose(float(np.sum(area.color_distribution)), 1.0, atol=1e-8, rtol=0.0)
