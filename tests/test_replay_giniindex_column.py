from __future__ import annotations

import json

import yaml

from scripts.run_headless import run_once
from src.config.loader import load_config
from src.config.schema import AppConfig
from src.replay.replay_server import ReplayModel


def test_replay_logs_and_exposes_area_giniindex(tmp_path):
    """End-to-end: headless logging writes per-area GiniIndex and replay exposes it in agent vars.

    Replay UI plumbing now uses snake_case internally.
    """
    conf = load_config('configs/toy.yaml')

    sim_cfg = conf.simulation
    sim_cfg = sim_cfg.model_copy(deep=True)
    sim_cfg.num_steps = 2
    sim_cfg.grid_interval = 1
    sim_cfg.store_grid = True

    run_dir = tmp_path / 'run_0'
    run_once(0, conf, out_dir=run_dir)

    pad = len(str(conf.simulation.num_steps))
    # step file should contain GiniIndex for each area
    step0 = json.loads((run_dir / 'steps' / f"step_{0:0{pad}d}.json").read_text())
    assert 'areas' in step0 and len(step0['areas']) > 0
    any_area = next(iter(step0['areas'].values()))
    assert 'GiniIndex' in any_area

    meta = yaml.safe_load((run_dir / 'meta.yaml').read_text())
    appcfg = AppConfig.model_validate(meta['config'])

    m = ReplayModel(appcfg=appcfg, run_dir=run_dir)
    m.step()

    df = m.datacollector.get_agent_vars_dataframe()
    assert 'gini_index' in df.columns
