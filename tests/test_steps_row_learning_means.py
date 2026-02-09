from __future__ import annotations

from pathlib import Path

import numpy as np

from src.logging.run_logger import RunLoggerV2
from tests.factory import create_test_model


def test_steps_row_includes_learning_means(tmp_path: Path) -> None:
    model, _ = create_test_model(num_agents=2, num_areas=1, num_colors=3)

    logger = RunLoggerV2(
        out_dir=tmp_path,
        run_seed=1,
        rule_idx=0,
        num_steps=1,
        store_grid=False,
    )

    # Provide pre-mutation area snapshots so _extract_steps_row can compute colors.
    step = 1
    for area in model.areas:
        logger._area_snapshots_by_step_area[(step, int(area.unique_id))] = {
            "area_color": np.asarray(area.color_distribution, dtype=np.float32),
            "elected_color": None,
        }

    row = logger._extract_steps_row(step=step, model=model)
    assert "mean_altruism" in row
    assert "mean_satisfaction" in row
