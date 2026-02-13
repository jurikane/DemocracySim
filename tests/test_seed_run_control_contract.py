from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.config.loader import load_config
from scripts.run_headless import run_once


def _read_meta_run_seed(out_dir: Path) -> int:
    payload = yaml.safe_load((out_dir / "meta.yaml").read_text(encoding="utf-8"))
    return int(payload["run"]["run_seed"])


def test_run_seed_oracle_is_base_seed_plus_run_id(tmp_path: Path) -> None:
    cfg = load_config("toy.yaml")
    cfg = cfg.model_copy(deep=True)
    cfg.simulation.num_steps = 1
    cfg.simulation.store_grid = False
    cfg.simulation.base_seed = 1234

    run_id = 7
    out_dir = tmp_path / "run"
    run_once(run_id, cfg, out_dir=out_dir)

    expected_seed = 1234 + run_id
    assert _read_meta_run_seed(out_dir) == expected_seed


def test_run_seed_metamorphic_run_id_changes_seed_by_one(tmp_path: Path) -> None:
    cfg = load_config("toy.yaml")
    cfg = cfg.model_copy(deep=True)
    cfg.simulation.num_steps = 1
    cfg.simulation.store_grid = False
    cfg.simulation.base_seed = 500

    out0 = tmp_path / "r0"
    out1 = tmp_path / "r1"
    run_once(0, cfg, out_dir=out0)
    run_once(1, cfg, out_dir=out1)

    s0 = _read_meta_run_seed(out0)
    s1 = _read_meta_run_seed(out1)
    assert s0 == 500
    assert s1 == 501
    assert s1 - s0 == 1


def test_run_seed_integration_logged_consistently_across_tables(tmp_path: Path) -> None:
    cfg = load_config("toy.yaml")
    cfg = cfg.model_copy(deep=True)
    cfg.simulation.num_steps = 2
    cfg.simulation.store_grid = False
    cfg.simulation.base_seed = 900

    out_dir = tmp_path / "run"
    run_once(3, cfg, out_dir=out_dir)
    expected_seed = 903

    assert _read_meta_run_seed(out_dir) == expected_seed

    for name in ("steps.parquet", "area_steps.parquet", "agents.parquet", "votes.parquet"):
        df = pd.read_parquet(out_dir / name)
        if df.empty:
            continue
        values = set(df["run_seed"].astype(int).tolist())
        assert values == {expected_seed}, f"{name} has inconsistent run_seed values: {values}"
