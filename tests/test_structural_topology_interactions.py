from __future__ import annotations

import numpy as np
import pandas as pd

from src.logging.run_logger import RunLoggerV2
from tests.factory import create_test_model


def _membership_counts(model) -> np.ndarray:
    counts = np.zeros((model.height, model.width), dtype=np.int32)
    for area in model.areas:
        for cell in area.cells:
            counts[int(cell.row), int(cell.col)] += 1
    return counts


def _run_one_logged_step(model, out_dir):
    logger = RunLoggerV2(
        out_dir=out_dir,
        run_seed=1,
        rule_idx=int(model.rule_idx),
        num_steps=1,
        store_grid=False,
    )
    logger.attach_to_model(model)
    logger.begin_step(1)
    model.step()
    logger.log_step(step=1, model=model, grid_snapshot=None)
    logger.end_step()
    logger.finalize()
    steps = pd.read_parquet(out_dir / "steps.parquet")
    area_steps = pd.read_parquet(out_dir / "area_steps.parquet")
    return steps, area_steps


def test_section_f_topology_interaction_partition_vs_overlap(tmp_path):
    # Topology A: exact partition (4x 5x5 areas on 10x10 grid).
    m_partition, _ = create_test_model(
        seed=1001,
        height=10,
        width=10,
        num_areas=4,
        av_area_height=5,
        av_area_width=5,
        area_size_variance=0.0,
        num_agents=20,
        num_colors=3,
    )
    mem_a = _membership_counts(m_partition)
    assert np.all(mem_a == 1)
    assert m_partition.no_overlap is True

    # Topology B: guaranteed overlap (2x 8x8 areas on 10x10 grid).
    # With torus and this size, the second area must overlap the first.
    m_overlap, _ = create_test_model(
        seed=1001,
        height=10,
        width=10,
        num_areas=2,
        av_area_height=8,
        av_area_width=8,
        area_size_variance=0.0,
        num_agents=20,
        num_colors=3,
    )
    mem_b = _membership_counts(m_overlap)
    assert np.any(mem_b > 1)
    assert m_overlap.no_overlap is False

    # Integration/logging check in both topologies.
    steps_a, area_steps_a = _run_one_logged_step(m_partition, tmp_path / "partition")
    steps_b, area_steps_b = _run_one_logged_step(m_overlap, tmp_path / "overlap")

    # One row in steps and one row per area in area_steps.
    assert len(steps_a) == 1
    assert len(steps_b) == 1
    assert len(area_steps_a) == m_partition.num_areas
    assert len(area_steps_b) == m_overlap.num_areas
    assert int(area_steps_a["area_id"].nunique()) == m_partition.num_areas
    assert int(area_steps_b["area_id"].nunique()) == m_overlap.num_areas

    # steps.turnout must equal resident-population weighted area turnout.
    ta = float(steps_a.iloc[0]["turnout"])
    tb = float(steps_b.iloc[0]["turnout"])
    ta_den = float(sum(int(a.num_agents) for a in m_partition.areas))
    tb_den = float(sum(int(a.num_agents) for a in m_overlap.areas))
    ta_mean = float(100.0 * area_steps_a["participants"].sum() / ta_den) if ta_den > 0 else 0.0
    tb_mean = float(100.0 * area_steps_b["participants"].sum() / tb_den) if tb_den > 0 else 0.0
    assert np.isfinite(ta) and np.isfinite(tb)
    np.testing.assert_allclose(ta, ta_mean, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(tb, tb_mean, rtol=0.0, atol=1e-5)
