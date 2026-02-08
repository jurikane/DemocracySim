from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config.loader import load_config
from scripts.run_headless import run_once
from src.logging.output_schema import (
    validate_steps_df,
    validate_area_steps_df,
    validate_agents_df,
    validate_votes_df,
    STEPS_TABLE,
    AREA_STEPS_TABLE,
    AGENTS_TABLE,
    VOTES_TABLE,
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
        run_dir / "votes.parquet",
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
    # static.json is required for schema v2 metadata.
    assert (v2_run_dir / "static.json").exists(), "static.json missing"

    # Required Parquet tables
    assert (v2_run_dir / "steps.parquet").exists(), "steps.parquet missing"
    assert (v2_run_dir / "area_steps.parquet").exists(), "area_steps.parquet missing"
    assert (v2_run_dir / "agents.parquet").exists(), "agents.parquet missing"
    assert (v2_run_dir / "votes.parquet").exists(), "votes.parquet missing"

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
    votes = pd.read_parquet(v2_run_dir / "votes.parquet")

    validate_steps_df(steps)
    validate_area_steps_df(area_steps)
    validate_agents_df(agents)
    validate_votes_df(votes)

    _assert_pk_unique(steps, STEPS_TABLE.primary_key, "steps")
    _assert_pk_unique(area_steps, AREA_STEPS_TABLE.primary_key, "area_steps")
    _assert_pk_unique(agents, AGENTS_TABLE.primary_key, "agents")
    _assert_pk_unique(votes, VOTES_TABLE.primary_key, "votes")


def test_votes_has_fixed_rank_columns(v2_run_dir: Path) -> None:
    """Contract: votes contains exactly the fixed 3-rank columns.

    This schema uses a wide layout to reduce row counts.

    Marked xfail until schema v2 logging is implemented.
    """
    reason = _schema_v2_missing_reason(v2_run_dir)
    if reason:
        pytest.xfail(reason)

    votes = pd.read_parquet(v2_run_dir / "votes.parquet")

    expected = {
        "rank_1_option_id",
        "rank_1_oppose_score",
        "rank_2_option_id",
        "rank_2_oppose_score",
        "rank_3_option_id",
        "rank_3_oppose_score",
    }
    missing = expected - set(votes.columns)
    assert not missing, f"Missing expected rank columns: {missing}"

    # Basic sanity: if option_id is present, oppose_score should be present.
    # (We don't enforce non-nullness because option space can be <3 in edge cases.)
    assert "rank_1_option_id" in votes.columns and "rank_1_oppose_score" in votes.columns


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


def test_schema_v2_step_indexing_is_one_based(v2_run_dir: Path) -> None:
    """Schema v2 must use 1-based step indexing for recorded snapshots.

    Note: a pre-election grid snapshot at step=0 is allowed for UI convenience.
    Parquet tables remain strictly 1-based.
    """
    reason = _schema_v2_missing_reason(v2_run_dir)
    if reason:
        pytest.xfail(reason)

    steps = pd.read_parquet(v2_run_dir / "steps.parquet")
    assert not steps.empty
    assert int(steps["step"].min()) == 1

    area_steps = pd.read_parquet(v2_run_dir / "area_steps.parquet")
    assert not area_steps.empty
    assert int(area_steps["step"].min()) == 1

    agents = pd.read_parquet(v2_run_dir / "agents.parquet")
    assert not agents.empty
    assert int(agents["step"].min()) == 1

    votes_path = v2_run_dir / "votes.parquet"
    assert votes_path.exists()
    votes = pd.read_parquet(votes_path)
    if not votes.empty:
        assert int(votes["step"].min()) == 1

    # Grid snapshots: if present, first recorded election snapshot idx must be 1.
    grids_dir = v2_run_dir / "grids"
    if grids_dir.exists():
        files = sorted(grids_dir.glob("grid_*.npy"))
        if files:
            def _idx(p: Path) -> int:
                suf = p.stem.rsplit("_", 1)[-1]
                return int(suf) if suf.isdigit() else 10**18

            idxs = sorted(_idx(p) for p in files)
            assert 1 in idxs, "v2 must write a grid snapshot for step=1"
            assert min([i for i in idxs if i >= 1]) == 1
