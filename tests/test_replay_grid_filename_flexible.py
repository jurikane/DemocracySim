from __future__ import annotations

import yaml
import numpy as np

from scripts.run_headless import run_once
from src.config.loader import load_config
from src.config.schema import AppConfig
from src.replay.replay_server import ReplayModel


def test_replay_loads_grid_with_flexible_filenames(tmp_path):
    """Regression: replay must load grid snapshots when filenames use padding derived from num_steps.

    Contract (schema v2):
      - Headless run writes grid snapshots using pad=len(str(num_steps)).
      - First recorded *election* grid is step=1 (grid_0001.npy).
      - An optional initial pre-election grid_0000.npy may exist for UI purposes.
      - ReplayModel loads grids by honoring static_v2.json step_indexing.grid_file.
    """
    cfg = load_config("configs/toy.yaml")
    cfg.simulation = cfg.simulation.model_copy(deep=True)
    cfg.simulation.num_steps = 30
    cfg.simulation.grid_interval = 1
    cfg.simulation.store_grid = True

    run_dir = tmp_path / "run_0"
    run_once(0, cfg, out_dir=run_dir)

    # Only padded filenames are required by the schema
    pad = len(str(cfg.simulation.num_steps))
    assert (run_dir / "grids" / f"grid_{1:0{pad}d}.npy").exists()
    # grid_0000.npy may also exist (pre-election snapshot)

    meta = yaml.safe_load((run_dir / "meta.yaml").read_text())
    appcfg = AppConfig.model_validate(meta["config"])

    m = ReplayModel(appcfg=appcfg, run_dir=run_dir)
    # Ensure at least one step applied and that colors are not all default 0
    colors = np.array([c.color for c in m.color_cells], dtype=int)
    assert colors.size > 0
    assert np.unique(colors).size > 1, "Replay grid colors did not update (still uniform)"
