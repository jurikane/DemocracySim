from __future__ import annotations

from scripts.run_headless import run_once
from src.config.loader import load_config
from src.replay.replay_server import ReplayModel
from tests.utils_meta import load_appcfg_from_meta


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

    appcfg = load_appcfg_from_meta(run_dir)

    m = ReplayModel(appcfg=appcfg, run_dir=run_dir)
    # ReplayModel starts at step 0 (grid only). Advance to step=1.
    m.step()
    df = m.datacollector.get_agent_vars_dataframe()
    assert 'gini_index' in df.columns
