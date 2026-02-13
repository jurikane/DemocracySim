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


def test_overlapping_topology_runs_multiple_steps_and_logs(tmp_path):
    # Guaranteed overlap: two 8x8 areas on a 10x10 torus/grid.
    model, _ = create_test_model(
        seed=1101,
        height=10,
        width=10,
        num_areas=2,
        av_area_height=8,
        av_area_width=8,
        area_size_variance=0.0,
        num_agents=30,
        num_colors=3,
        max_steps=3,
    )

    membership = _membership_counts(model)
    assert np.any(membership > 1), "Expected real overlap in this topology."
    assert model.no_overlap is False

    logger = RunLoggerV2(
        out_dir=tmp_path,
        run_seed=1,
        rule_idx=int(model.rule_idx),
        num_steps=3,
        store_grid=False,
    )
    logger.attach_to_model(model)

    for step in (1, 2, 3):
        logger.begin_step(step)
        model.step()
        logger.log_step(step=step, model=model, grid_snapshot=None)
        logger.end_step()

        # Runtime invariants while overlapping mode is active.
        assert np.isfinite(model.global_color_dst).all()
        np.testing.assert_allclose(float(np.sum(model.global_color_dst)), 1.0, atol=1e-8, rtol=0.0)
        for area in model.areas:
            assert np.isfinite(area.color_distribution).all()
            np.testing.assert_allclose(float(np.sum(area.color_distribution)), 1.0, atol=1e-8, rtol=0.0)

    logger.finalize()

    steps = pd.read_parquet(tmp_path / "steps.parquet").sort_values("step")
    area_steps = pd.read_parquet(tmp_path / "area_steps.parquet").sort_values(["step", "area_id"])

    # Integration expectations in overlap mode.
    assert len(steps) == 3
    assert len(area_steps) == 3 * model.num_areas
    assert int(area_steps["area_id"].nunique()) == model.num_areas
    assert np.isfinite(steps["turnout"]).all()
    assert np.isfinite(area_steps["turnout"]).all()

