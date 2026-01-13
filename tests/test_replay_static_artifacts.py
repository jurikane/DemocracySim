from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.run_headless import run_once
from src.config.loader import load_config


def test_replay_static_includes_borders_and_voter_counts(tmp_path):
    """Schema v1 completeness regression:

    New runs must write static artifacts needed for replay + analysis:
      - static.json includes total_voters + voters_per_area
      - area_borders.npy exists and has shape (H, W)

    This test generates a tiny run to avoid depending on checked-in sample data
    that may have been produced before these fields existed.
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

    run_once(0, conf, sim_cfg, out_dir=run_dir)

    static = json.loads((run_dir / 'static.json').read_text())
    assert static.get('format_version') == 1
    assert 'total_voters' in static
    assert isinstance(static['total_voters'], int)
    assert 'voters_per_area' in static
    assert isinstance(static['voters_per_area'], dict)

    borders_path = run_dir / 'area_borders.npy'
    assert borders_path.exists()
    arr = np.load(str(borders_path))
    assert arr.ndim == 2
    assert arr.shape == (int(static['height']), int(static['width']))
