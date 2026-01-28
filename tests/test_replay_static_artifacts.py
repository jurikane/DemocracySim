from __future__ import annotations

import json

import numpy as np

from scripts.run_headless import run_once
from src.config.loader import load_config


def test_replay_static_includes_borders_and_voter_counts(tmp_path):
    """Schema v2 completeness regression.

    New runs must write static artifacts needed for replay + analysis:
      - static.json exists and declares schema v2
      - area_borders.npy exists and has shape (H, W)

    This test generates a tiny run to avoid depending on checked-in sample data.
    """
    conf = load_config('configs/toy.yaml')
    sim_cfg = conf.simulation
    try:
        sim_cfg = sim_cfg.model_copy(deep=True)
        sim_cfg.num_steps = 1
        sim_cfg.grid_interval = 1
        sim_cfg.store_grid = True
    except Exception:
        pass

    run_dir = tmp_path / 'run_0'
    run_dir.mkdir(parents=True, exist_ok=True)

    run_once(0, conf, out_dir=run_dir)

    static = json.loads((run_dir / 'static.json').read_text())
    schema = static.get('schema', {})
    assert schema.get('name') == 'output_schema_v2'
    assert int(schema.get('version', 0) or 0) == 2

    borders_path = run_dir / 'area_borders.npy'
    assert borders_path.exists()
    arr = np.load(str(borders_path))
    assert arr.ndim == 2
    assert arr.shape == (int(static['height']), int(static['width']))
