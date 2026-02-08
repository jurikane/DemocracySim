from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config.loader import load_config
from src.model_setup import make_model
from src.logging.run_logger import RunLoggerV2


def test_area_steps_use_pre_mutation_snapshot(tmp_path: Path) -> None:
    """Ensure area_steps rows use pre-mutation snapshots when provided."""
    cfg = load_config("toy.yaml")
    cfg_for_run = cfg.model_copy(deep=True)
    cfg_for_run.simulation.num_steps = 1
    cfg_for_run.simulation.store_grid = False

    model = make_model(cfg_for_run.model)
    rule_idx = int(cfg_for_run.model.rule_idx)
    logger = RunLoggerV2(out_dir=tmp_path, run_seed=1, rule_idx=rule_idx, num_steps=1, store_grid=False)

    logger.attach_to_model(model)
    logger.begin_step(1)
    model.step()

    # Inject a pre-mutation snapshot (should be used for area_steps).
    area = next(a for a in model.areas if a is not None)
    num_colors = int(model.num_colors)
    snapshot_area_color = [0.0] * num_colors
    snapshot_elected_color = list(range(num_colors))
    logger._area_snapshots_by_step_area[(1, int(area.unique_id))] = {
        "area_color": snapshot_area_color,
        "elected_color": snapshot_elected_color,
        "turnout": 12.5,
        "participants": 1,
        "eligible_voters": int(area.num_agents),
        "dist_to_reality": 0.123,
    }

    logger.log_step(step=1, model=model, grid_snapshot=None)
    logger.end_step()

    rows = [r for r in logger._area_steps_rows if int(r["area_id"]) == int(area.unique_id)]
    assert rows, "Expected at least one area_steps row"
    row = rows[0]

    assert np.isclose(float(row["turnout"]), 12.5)
    assert int(row["participants"]) == 1
    assert np.isclose(float(row["dist_to_reality"]), 0.123)
    # area_color_* should match the pre-mutation snapshot, not the current area state
    for i in range(num_colors):
        assert np.isclose(
            float(row[f"area_color_{i}"]),
            float(snapshot_area_color[i]),
        )
