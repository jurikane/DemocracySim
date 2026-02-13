from __future__ import annotations

import json
from math import factorial

import numpy as np
import pytest

from src.logging.run_logger import RunLoggerV2
from tests.factory import create_test_model


def test_num_personality_groups_validation_fail_loud():
    with pytest.raises(ValueError):
        create_test_model(num_personality_groups=0)
    with pytest.raises(ValueError):
        create_test_model(num_personality_groups=-1)
    with pytest.raises(ValueError):
        create_test_model(num_personality_groups=1.5)
    with pytest.raises(ValueError):
        create_test_model(num_personality_groups=True)

    # For num_colors=3 max unique groups is 3! = 6.
    with pytest.raises(ValueError):
        create_test_model(num_colors=3, num_personality_groups=7)


def test_num_personality_groups_oracle_shape_and_uniqueness():
    n = 5
    model, _ = create_test_model(
        seed=1501,
        num_colors=4,
        num_personality_groups=n,
        num_agents=30,
    )
    groups = np.asarray(model.personality_groups, dtype=np.int64)
    assert groups.shape == (n, int(model.num_colors))
    assert len(set(map(tuple, groups.tolist()))) == n

    # Each personality group must be a valid permutation of color ids.
    expected = list(range(int(model.num_colors)))
    for row in groups:
        assert sorted(row.tolist()) == expected

    # Global distribution should align with number of groups.
    pgd = np.asarray(model.personality_group_distribution, dtype=np.float64)
    assert pgd.shape == (n,)
    np.testing.assert_allclose(float(np.sum(pgd)), 1.0, rtol=0.0, atol=1e-12)


def test_num_personality_groups_metamorphic_distribution_length_tracks_knob():
    m1, _ = create_test_model(seed=1502, num_colors=4, num_personality_groups=3, num_agents=30)
    m2, _ = create_test_model(seed=1502, num_colors=4, num_personality_groups=6, num_agents=30)
    assert len(m1.personality_groups) == 3
    assert len(m2.personality_groups) == 6
    assert len(m1.personality_group_distribution) == 3
    assert len(m2.personality_group_distribution) == 6


def test_num_personality_groups_integration_static_metadata_lengths(tmp_path):
    n = 4
    model, _ = create_test_model(
        seed=1503,
        num_colors=4,
        num_personality_groups=n,
        num_areas=2,
        num_agents=20,
    )
    logger = RunLoggerV2(
        out_dir=tmp_path,
        run_seed=1,
        rule_idx=int(model.rule_idx),
        num_steps=1,
        store_grid=False,
    )
    logger.write_static(model)

    payload = json.loads((tmp_path / "static.json").read_text(encoding="utf-8"))
    pg_info = payload["personality_group_info"]
    global_dist = pg_info["global_distribution"]
    assert len(global_dist) == n

    area_info = pg_info["areas"]
    assert isinstance(area_info, dict)
    assert len(area_info) == int(model.num_areas)
    for entry in area_info.values():
        dist = entry["personality_group_distribution"]
        assert len(dist) == n
