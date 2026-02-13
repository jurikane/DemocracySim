from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_headless import run_once
from src.config.loader import load_config
from src.model_setup import make_model
from src.replay.replay_server import ReplayModel


def _headless_steps(run_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(run_dir / "steps.parquet").sort_values("step").reset_index(drop=True)


def _live_steps_from_model(cfg, *, run_seed: int, num_steps: int) -> pd.DataFrame:
    cfg_live = cfg.model_copy(deep=True)
    cfg_live.model.seed = int(run_seed)
    model = make_model(cfg_live.model)
    for _ in range(int(num_steps)):
        model.step()
    # DataCollector includes an initial row from model init; keep election-time rows only.
    return model.datacollector.get_model_vars_dataframe().tail(int(num_steps)).reset_index(drop=True)


def _replay_steps(run_dir: Path, cfg) -> pd.DataFrame:
    m = ReplayModel(appcfg=cfg, run_dir=run_dir)
    for _ in range(len(m.data)):
        m.step()
    return m.datacollector.get_model_vars_dataframe().reset_index(drop=True)


def test_live_headless_replay_equivalence_same_seed_key_series(tmp_path: Path) -> None:
    """P0 contract: live/headless/replay must agree on key model series for same seed."""
    cfg = load_config("toy.yaml").model_copy(deep=True)
    cfg.simulation.num_steps = 4
    cfg.simulation.store_grid = True
    cfg.simulation.grid_interval = 1
    cfg.simulation.base_seed = 9100

    run_dir = tmp_path / "run"
    run_once(run_id=0, cfg=cfg, out_dir=run_dir)

    run_seed = int(cfg.simulation.base_seed) + 0
    n = int(cfg.simulation.num_steps)

    s_headless = _headless_steps(run_dir)
    s_live = _live_steps_from_model(cfg, run_seed=run_seed, num_steps=n)
    s_replay = _replay_steps(run_dir, cfg)

    assert len(s_headless) == len(s_live) == len(s_replay) == n

    keys = ["collective_assets", "turnout", "gini_index"]
    keys.extend(sorted(c for c in s_headless.columns if c.startswith("color_")))

    for col in keys:
        assert col in s_live.columns, f"{col} missing in live DataCollector"
        assert col in s_replay.columns, f"{col} missing in replay DataCollector"
        atol = 1e-6 if col != "collective_assets" else 1e-3
        np.testing.assert_allclose(
            s_live[col].to_numpy(dtype=float),
            s_headless[col].to_numpy(dtype=float),
            rtol=0,
            atol=atol,
        )
        np.testing.assert_allclose(
            s_replay[col].to_numpy(dtype=float),
            s_headless[col].to_numpy(dtype=float),
            rtol=0,
            atol=atol,
        )
