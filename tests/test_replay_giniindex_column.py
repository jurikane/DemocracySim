from __future__ import annotations

import yaml

from scripts.run_headless import run_once
from src.config.loader import load_config
from src.config.schema import AppConfig
from src.replay.replay_server import ReplayModel


def test_replay_logs_and_exposes_area_giniindex(tmp_path):
    """End-to-end: schema v2 logging writes per-area gini_index and replay exposes it."""
    conf = load_config('configs/toy.yaml')

    sim_cfg = conf.simulation
    sim_cfg = sim_cfg.model_copy(deep=True)
    sim_cfg.num_steps = 2
    sim_cfg.grid_interval = 1
    sim_cfg.store_grid = True

    run_dir = tmp_path / 'run_0'
    run_once(0, conf, out_dir=run_dir)

    # Parquet must contain per-area gini_index
    import pandas as pd
    area_steps = pd.read_parquet(run_dir / 'area_steps.parquet')
    assert not area_steps.empty
    step1 = area_steps[area_steps['step'].astype(int) == 1]
    assert not step1.empty
    assert 'gini_index' in step1.columns

    meta = yaml.safe_load((run_dir / 'meta.yaml').read_text())
    appcfg = AppConfig.model_validate(meta['config'])

    m = ReplayModel(appcfg=appcfg, run_dir=run_dir)
    # ReplayModel starts at step 0 (grid only). Advance to step=1.
    m.step()
    df = m.datacollector.get_agent_vars_dataframe()
    assert 'gini_index' in df.columns
