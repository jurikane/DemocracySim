from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config.loader import load_config
from scripts.run_headless import run_once
from src.logging.output_schema_v2 import (
    validate_steps_df,
    validate_area_steps_df,
    validate_agents_df,
    validate_votes_topk_df,
    STEPS_TABLE,
    AREA_STEPS_TABLE,
    AGENTS_TABLE,
    VOTES_TOPK_TABLE,
)


pytestmark = pytest.mark.phase1


@pytest.fixture()
def v2_run_dir(tmp_path: Path) -> Path:
    """Create a tiny headless run in a temp directory.

    This is a *contract test* for schema v2.

    NOTE: Expected to FAIL until logging is migrated to schema v2.
    """
    cfg = load_config("toy.yaml")

    # Ensure a tiny run for speed.
    # We avoid mutating the shared config object by deep-copying.
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


def _schema_v2_missing_reason(run_dir: Path) -> str:
    """Return a short reason string if schema v2 artifacts are not present."""
    required = [
        run_dir / "steps.parquet",
        run_dir / "area_steps.parquet",
        run_dir / "agents.parquet",
        run_dir / "votes_topk.parquet",
    ]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        return f"schema v2 parquet artifacts not present yet: missing {missing}"
    return ""


def test_schema_v2_required_artifacts_exist(v2_run_dir: Path) -> None:
    """Contract: schema v2 run dir contains the required files.

    Marked xfail until schema v2 logging is implemented.
    """
    reason = _schema_v2_missing_reason(v2_run_dir)
    if reason:
        pytest.xfail(reason)

    # Core run metadata
    assert (v2_run_dir / "meta.yaml").exists(), "meta.yaml missing"
    assert (v2_run_dir / "static.json").exists(), "static.json missing"

    # Required Parquet tables
    assert (v2_run_dir / "steps.parquet").exists(), "steps.parquet missing"
    assert (v2_run_dir / "area_steps.parquet").exists(), "area_steps.parquet missing"
    assert (v2_run_dir / "agents.parquet").exists(), "agents.parquet missing"
    assert (v2_run_dir / "votes_topk.parquet").exists(), "votes_topk.parquet missing"

    # Required grids
    grids_dir = v2_run_dir / "grids"
    assert grids_dir.exists(), "grids/ missing"
    # For 3 steps, we expect at least grid_000.npy naming based on replay pad.
    # We don't enforce exact zero-padding here, just presence of at least 1 grid.
    assert any(grids_dir.glob("grid_*.npy")), "no grid_*.npy snapshots found"


def test_schema_v2_parquets_validate_and_keys_unique(v2_run_dir: Path) -> None:
    """Contract: Parquet files validate with schema v2 validators and PKs are unique.

    Marked xfail until schema v2 logging is implemented.
    """
    reason = _schema_v2_missing_reason(v2_run_dir)
    if reason:
        pytest.xfail(reason)

    steps = pd.read_parquet(v2_run_dir / "steps.parquet")
    area_steps = pd.read_parquet(v2_run_dir / "area_steps.parquet")
    agents = pd.read_parquet(v2_run_dir / "agents.parquet")
    votes = pd.read_parquet(v2_run_dir / "votes_topk.parquet")

    validate_steps_df(steps)
    validate_area_steps_df(area_steps)
    validate_agents_df(agents)
    validate_votes_topk_df(votes)

    _assert_pk_unique(steps, STEPS_TABLE.primary_key, "steps")
    _assert_pk_unique(area_steps, AREA_STEPS_TABLE.primary_key, "area_steps")
    _assert_pk_unique(agents, AGENTS_TABLE.primary_key, "agents")
    _assert_pk_unique(votes, VOTES_TOPK_TABLE.primary_key, "votes_topk")


def test_votes_topk_rank_within_bounds(v2_run_dir: Path) -> None:
    """Contract: votes_topk.rank is within 1...k.
    If k is known from config use it, otherwise assume k=3.

    Marked xfail until schema v2 logging is implemented.
    """
    reason = _schema_v2_missing_reason(v2_run_dir)
    if reason:
        pytest.xfail(reason)

    votes = pd.read_parquet(v2_run_dir / "votes_topk.parquet")

    # If no explicit k exists in config, default to 3 (project contract).
    cfg = load_config("toy.yaml")
    k = int(getattr(getattr(cfg, "simulation", None), "k", 3) or 3)
    if k <= 0:
        k = 3

    assert "rank" in votes.columns
    assert votes["rank"].min() >= 1
    assert votes["rank"].max() <= k


def test_grids_are_loadable_numpy_arrays(v2_run_dir: Path) -> None:
    """Grids are part of both replay schema v1 and schema v2.
    This test should pass always.
    """
    grids_dir = v2_run_dir / "grids"
    files = sorted(grids_dir.glob("grid_*.npy"))
    assert files, "No grid snapshots to load"

    arr0 = np.load(str(files[0]))
    assert isinstance(arr0, np.ndarray)
    assert arr0.ndim == 2
