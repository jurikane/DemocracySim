import pytest

from tests.factory import create_test_model


def test_num_areas_validation_requires_int_ge_0():
    with pytest.raises(ValueError):
        create_test_model(num_areas=-1)
    with pytest.raises(ValueError):
        create_test_model(num_areas=1.5)
    with pytest.raises(ValueError):
        create_test_model(num_areas=True)


def test_num_areas_validation_rejects_more_than_grid_slots():
    with pytest.raises(ValueError):
        create_test_model(height=3, width=3, num_areas=10)


def test_num_areas_zero_creates_no_areas():
    model, _ = create_test_model(
        height=8, width=8,
        num_areas=0,
        av_area_height=4, av_area_width=4,
        area_size_variance=0.0,
        seed=700,
    )
    assert model.num_areas == 0
    assert len(model.areas) == 0


def test_num_areas_count_matches_requested_value():
    n = 5
    model, _ = create_test_model(
        height=10, width=10,
        num_areas=n,
        av_area_height=5, av_area_width=5,
        area_size_variance=0.0,
        seed=701,
    )
    assert model.num_areas == n
    assert len(model.areas) == n
    assert all(a is not None for a in model.areas)

