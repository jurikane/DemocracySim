from __future__ import annotations

import numpy as np

from src.logging.run_logger import RunLoggerV2
from tests.factory import create_test_model


def _membership_counts(model) -> np.ndarray:
    counts = np.zeros((model.height, model.width), dtype=np.int32)
    for area in model.areas:
        for cell in area.cells:
            counts[int(cell.row), int(cell.col)] += 1
    return counts


def _grid_distribution_from_cells(model) -> np.ndarray:
    counts = np.zeros(model.num_colors, dtype=np.float64)
    for c in model.color_cells:
        counts[int(c.color)] += 1.0
    return counts / float(len(model.color_cells))


def test_no_overlap_oracle_true_implies_exact_partition():
    model, _ = create_test_model(
        seed=901,
        height=10,
        width=10,
        num_areas=4,
        av_area_height=5,
        av_area_width=5,
        area_size_variance=0.0,
    )
    assert model.no_overlap is True
    membership = _membership_counts(model)
    assert np.all(membership <= 1), "no_overlap=True must imply no shared cells."


def test_no_overlap_oracle_false_means_actual_overlap_exists():
    model, _ = create_test_model(
        seed=902,
        height=10,
        width=10,
        num_areas=5,
        av_area_height=5,
        av_area_width=5,
        area_size_variance=0.0,
    )
    assert model.no_overlap is False
    membership = _membership_counts(model)
    assert np.any(membership > 1), "no_overlap=False must indicate at least one overlapping cell."


def test_no_overlap_metamorphic_by_topology_change():
    base = dict(
        seed=903,
        height=10,
        width=10,
        av_area_height=5,
        av_area_width=5,
        area_size_variance=0.0,
    )
    m_partition, _ = create_test_model(**base, num_areas=4)
    m_non_partition, _ = create_test_model(**base, num_areas=5)
    assert m_partition.no_overlap is True
    assert m_non_partition.no_overlap is False


def test_disjoint_with_gaps_still_reports_no_overlap_true():
    model, _ = create_test_model(
        seed=905,
        height=10,
        width=10,
        num_areas=1,
        av_area_height=5,
        av_area_width=5,
        area_size_variance=0.0,
    )
    membership = _membership_counts(model)
    assert np.any(membership == 0), "This setup should leave uncovered cells (gaps)."
    assert np.all(membership <= 1), "Setup must be disjoint."
    assert model.no_overlap is True


def test_overlap_semantics_integration_steps_color_matches_grid_for_partition(tmp_path):
    model, _ = create_test_model(
        seed=904,
        height=10,
        width=10,
        num_areas=4,
        av_area_height=5,
        av_area_width=5,
        area_size_variance=0.0,
        num_colors=3,
        num_agents=8,
    )
    assert model.no_overlap is True

    logger = RunLoggerV2(
        out_dir=tmp_path,
        run_seed=1,
        rule_idx=int(model.rule_idx),
        num_steps=1,
        store_grid=False,
    )

    step = 1
    for area in model.areas:
        logger._area_snapshots_by_step_area[(step, int(area.unique_id))] = {
            "area_color": np.asarray(area.color_distribution, dtype=np.float32),
            "elected_color": list(range(int(model.num_colors))),
            "turnout": float(area.voter_turnout),
            "participants": 0,
            "eligible_voters": int(area.num_agents),
            "dist_to_reality": 0.0,
        }

    row = logger._extract_steps_row(step=step, model=model)
    logged = np.array([float(row[f"color_{i}"]) for i in range(int(model.num_colors))], dtype=np.float64)
    expected = _grid_distribution_from_cells(model)
    np.testing.assert_allclose(logged, expected, rtol=0.0, atol=1e-6)
