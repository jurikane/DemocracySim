from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.run_headless import run_once
from src.config.loader import load_config


def _run_with_steps(tmp_path: Path, *, num_steps: int, seed: int) -> Path:
    cfg = load_config("toy.yaml").model_copy(deep=True)
    cfg.simulation.num_steps = int(num_steps)
    cfg.simulation.store_grid = False
    cfg.simulation.grid_interval = 1
    cfg.simulation.base_seed = int(seed)
    out_dir = tmp_path / f"run_s{num_steps}"
    run_once(run_id=0, cfg=cfg, out_dir=out_dir)
    return out_dir


def test_num_steps_oracle_row_counts_and_step_range(tmp_path: Path) -> None:
    out = _run_with_steps(tmp_path, num_steps=4, seed=2101)
    steps = pd.read_parquet(out / "steps.parquet")
    area_steps = pd.read_parquet(out / "area_steps.parquet")
    agents = pd.read_parquet(out / "agents.parquet")

    assert len(steps) == 4
    assert int(steps["step"].min()) == 1
    assert int(steps["step"].max()) == 4

    num_areas = int(area_steps["area_id"].nunique())
    assert len(area_steps) == 4 * num_areas
    assert int(area_steps["step"].min()) == 1
    assert int(area_steps["step"].max()) == 4

    num_agents = int(agents["agent_id"].nunique())
    assert len(agents) == 4 * num_agents
    assert int(agents["step"].min()) == 1
    assert int(agents["step"].max()) == 4


def test_num_steps_metamorphic_prefix_invariance(tmp_path: Path) -> None:
    """With same seed and config, a longer run must have the shorter run as prefix."""
    out_short = _run_with_steps(tmp_path, num_steps=2, seed=2102)
    out_long = _run_with_steps(tmp_path, num_steps=5, seed=2102)

    s_short = pd.read_parquet(out_short / "steps.parquet").sort_values("step").reset_index(drop=True)
    s_long = pd.read_parquet(out_long / "steps.parquet").sort_values("step").reset_index(drop=True)
    pd.testing.assert_frame_equal(s_short, s_long.head(2), check_dtype=False)

    a_short = pd.read_parquet(out_short / "area_steps.parquet").sort_values(["step", "area_id"]).reset_index(drop=True)
    a_long = pd.read_parquet(out_long / "area_steps.parquet").sort_values(["step", "area_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a_short, a_long.head(len(a_short)), check_dtype=False)


def test_num_steps_integration_meta_and_parquet_consistency(tmp_path: Path) -> None:
    out = _run_with_steps(tmp_path, num_steps=3, seed=2103)
    steps = pd.read_parquet(out / "steps.parquet")
    area_steps = pd.read_parquet(out / "area_steps.parquet")
    agents = pd.read_parquet(out / "agents.parquet")
    votes = pd.read_parquet(out / "votes.parquet")

    assert set(steps["step"].tolist()) == {1, 2, 3}
    assert set(area_steps["step"].unique().tolist()) == {1, 2, 3}
    assert set(agents["step"].unique().tolist()) == {1, 2, 3}
    if not votes.empty:
        assert set(votes["step"].unique().tolist()).issubset({1, 2, 3})
