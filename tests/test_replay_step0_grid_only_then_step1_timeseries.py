from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.run_headless import run_once
from src.config.loader import load_config
from src.replay.replay_server import ReplayModel


@pytest.mark.phase1
def test_replay_starts_with_step0_grid_only_then_step1_timeseries(tmp_path: Path) -> None:
    """Regression: replay must start at step 0 with grid-only, then populate series at step 1.

    Contract (schema v2 hard-cut):
    - Headless writes a pre-election grid snapshot as grids/grid_0000.npy.
    - ReplayModel __init__ applies that grid without adding any model/area time-series rows.
    - First call to ReplayModel.step() advances to recorded step=1:
      - scheduler.steps becomes 1
      - datacollector contains model vars + area vars (non-empty)
    """

    cfg = load_config("configs/toy.yaml").model_copy(deep=True)
    cfg.simulation.num_steps = 2
    cfg.simulation.grid_interval = 1
    cfg.simulation.store_grid = True

    run_dir = tmp_path / "run_0"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_once(0, cfg, out_dir=run_dir)

    # Grid 0 must exist (pre-election snapshot)
    pad = len(str(int(cfg.simulation.num_steps)))
    g0_path = run_dir / "grids" / f"grid_{0:0{pad}d}.npy"
    assert g0_path.exists(), "Expected pre-election grid_0000.npy to exist"

    m = ReplayModel(appcfg=cfg, run_dir=run_dir)

    # UI step display uses scheduler.steps; it must start at 0.
    assert int(m.scheduler.steps) == 0

    # DataCollector should be empty at step 0 (grid-only)
    assert m.datacollector.get_model_vars_dataframe().empty
    assert m.datacollector.get_agent_vars_dataframe().empty

    # Ensure grid applied is not all zeros (sanity)
    colors0 = np.array([c.color for c in m.color_cells], dtype=int)
    assert colors0.size == m.height * m.width
    assert np.unique(colors0).size >= 1

    # First step should materialize step=1 and populate time series
    m.step()
    assert int(m.scheduler.steps) == 1

    model_df = m.datacollector.get_model_vars_dataframe()
    area_df = m.datacollector.get_agent_vars_dataframe()
    assert not model_df.empty
    assert not area_df.empty

    # Must contain core snake_case model series
    for col in ("collective_assets", "turnout", "gini_index"):
        assert col in model_df.columns
