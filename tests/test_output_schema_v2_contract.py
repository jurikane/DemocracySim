from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_headless import run_once
from src.config.loader import load_config
from src.logging.output_schema import (
    AGENTS_TABLE,
    AREA_STEPS_TABLE,
    STEPS_TABLE,
    VOTES_TABLE,
    validate_agents_df,
    validate_area_steps_df,
    validate_steps_df,
    validate_votes_df,
)


pytestmark = pytest.mark.phase1


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    cfg = load_config("toy.yaml")
    cfg_for_run = cfg.model_copy(deep=True)
    cfg_for_run.simulation.num_steps = 3
    cfg_for_run.simulation.runs = 1
    cfg_for_run.simulation.grid_interval = 1
    cfg_for_run.simulation.store_grid = True

    out_dir = tmp_path / "run"
    run_once(0, cfg_for_run, out_dir=out_dir)
    return out_dir


def _assert_pk_unique(df: pd.DataFrame, pk: tuple[str, ...], table_name: str) -> None:
    dup = df.duplicated(list(pk))
    if dup.any():
        examples = df.loc[dup, list(pk)].head(5).to_dict("records")
        raise AssertionError(f"{table_name}: primary key has duplicates. Examples: {examples}")


def test_required_artifacts_exist(run_dir: Path) -> None:
    required = (
        "meta.yaml",
        "static.json",
        "steps.parquet",
        "area_steps.parquet",
        "agents.parquet",
        "votes.parquet",
    )
    for name in required:
        assert (run_dir / name).exists(), f"{name} missing"

    grids_dir = run_dir / "grids"
    assert grids_dir.exists(), "grids/ missing"
    assert any(grids_dir.glob("grid_*.npy")), "no grid_*.npy snapshots found"


def test_parquets_validate_and_keys_unique(run_dir: Path) -> None:
    steps = pd.read_parquet(run_dir / "steps.parquet")
    area_steps = pd.read_parquet(run_dir / "area_steps.parquet")
    agents = pd.read_parquet(run_dir / "agents.parquet")
    votes = pd.read_parquet(run_dir / "votes.parquet")

    validate_steps_df(steps)
    validate_area_steps_df(area_steps)
    validate_agents_df(agents)
    validate_votes_df(votes)

    _assert_pk_unique(steps, STEPS_TABLE.primary_key, "steps")
    _assert_pk_unique(area_steps, AREA_STEPS_TABLE.primary_key, "area_steps")
    _assert_pk_unique(agents, AGENTS_TABLE.primary_key, "agents")
    _assert_pk_unique(votes, VOTES_TABLE.primary_key, "votes")


def test_votes_has_fixed_rank_columns(run_dir: Path) -> None:
    votes = pd.read_parquet(run_dir / "votes.parquet")
    expected = {
        "voted_altruistically",
        "rank_1_option_id",
        "rank_1_oppose_score",
        "rank_2_option_id",
        "rank_2_oppose_score",
        "rank_3_option_id",
        "rank_3_oppose_score",
    }
    missing = expected - set(votes.columns)
    assert not missing, f"Missing expected rank columns: {sorted(missing)}"


def test_grids_are_loadable_numpy_arrays(run_dir: Path) -> None:
    files = sorted((run_dir / "grids").glob("grid_*.npy"))
    assert files, "No grid snapshots to load"
    arr = np.load(str(files[0]))
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 2


def test_step_indexing_is_one_based(run_dir: Path) -> None:
    steps = pd.read_parquet(run_dir / "steps.parquet")
    area_steps = pd.read_parquet(run_dir / "area_steps.parquet")
    agents = pd.read_parquet(run_dir / "agents.parquet")
    votes = pd.read_parquet(run_dir / "votes.parquet")

    assert int(steps["step"].min()) == 1
    assert int(area_steps["step"].min()) == 1
    assert int(agents["step"].min()) == 1
    if not votes.empty:
        assert int(votes["step"].min()) == 1

