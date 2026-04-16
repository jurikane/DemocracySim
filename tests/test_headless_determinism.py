from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config.loader import load_config
from scripts.run_headless import run_once


def _assert_runs_identical(out_a: Path, out_b: Path, *, pad: int) -> None:
    def _sort(df: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
        keys = [c for c in candidates if c in df.columns]
        if not keys:
            return df.reset_index(drop=True)
        return df.sort_values(keys).reset_index(drop=True)

    for rel in [
        Path("static.json"),
        Path("meta.yaml"),
        Path("steps.parquet"),
        Path("area_steps.parquet"),
        Path("agents.parquet"),
        Path("votes.parquet"),
    ]:
        assert (out_a / rel).exists(), rel
        assert (out_b / rel).exists(), rel

    # Compare complete tables (sorted by stable PK-like columns).
    s_a = _sort(pd.read_parquet(out_a / "steps.parquet"), ["step"])
    s_b = _sort(pd.read_parquet(out_b / "steps.parquet"), ["step"])
    pd.testing.assert_frame_equal(s_a, s_b, check_dtype=False)

    a_a = _sort(pd.read_parquet(out_a / "area_steps.parquet"), ["step", "area_id"])
    a_b = _sort(pd.read_parquet(out_b / "area_steps.parquet"), ["step", "area_id"])
    pd.testing.assert_frame_equal(a_a, a_b, check_dtype=False)

    g_a = _sort(pd.read_parquet(out_a / "agents.parquet"), ["step", "agent_id"])
    g_b = _sort(pd.read_parquet(out_b / "agents.parquet"), ["step", "agent_id"])
    pd.testing.assert_frame_equal(g_a, g_b, check_dtype=False)

    v_a = _sort(pd.read_parquet(out_a / "votes.parquet"), ["step", "area_id", "agent_id"])
    v_b = _sort(pd.read_parquet(out_b / "votes.parquet"), ["step", "area_id", "agent_id"])
    pd.testing.assert_frame_equal(v_a, v_b, check_dtype=False)

    # Compare all stored grid snapshots.
    grids_a = sorted((out_a / "grids").glob("grid_*.npy"))
    grids_b = sorted((out_b / "grids").glob("grid_*.npy"))
    assert [p.name for p in grids_a] == [p.name for p in grids_b]
    for pa, pb in zip(grids_a, grids_b):
        np.testing.assert_array_equal(np.load(pa), np.load(pb))

    # Explicitly keep fast check for first two schema-v2 steps.
    g1_a = out_a / "grids" / f"grid_{1:0{pad}d}.npy"
    g1_b = out_b / "grids" / f"grid_{1:0{pad}d}.npy"
    if g1_a.exists() and g1_b.exists():
        np.testing.assert_array_equal(np.load(g1_a), np.load(g1_b))
    g2_a = out_a / "grids" / f"grid_{2:0{pad}d}.npy"
    g2_b = out_b / "grids" / f"grid_{2:0{pad}d}.npy"
    if g2_a.exists() and g2_b.exists():
        np.testing.assert_array_equal(np.load(g2_a), np.load(g2_b))


def test_headless_same_seed_produces_identical_first_steps(tmp_path: Path):
    """Critical determinism check for logged runs.

    Contract (schema v2): with same config + run_id (=> same run_seed), the first few
    logged artifacts should be identical in meaning.

    We compare parquet content (rows) and a couple of grid snapshots.
    """
    cfg = load_config("toy.yaml")
    cfg = cfg.model_copy(deep=True)
    cfg.simulation.num_steps = 3
    cfg.simulation.store_grid = True
    cfg.simulation.grid_interval = 1
    cfg.simulation.base_seed = 777
    pad = len(str(cfg.simulation.num_steps))

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    run_once(0, cfg, out_dir=out_a)
    run_once(0, cfg, out_dir=out_b)
    _assert_runs_identical(out_a, out_b, pad=pad)


@pytest.mark.parametrize("cfg_name", ["toy.yaml"])
def test_headless_same_seed_reproducible_across_config_variants(tmp_path: Path, cfg_name: str) -> None:
    """Reproducibility must hold across different config profiles, including debug-enabled ones."""
    cfg = load_config(cfg_name)
    cfg = cfg.model_copy(deep=True)
    cfg.simulation.num_steps = 3
    cfg.simulation.store_grid = True
    cfg.simulation.grid_interval = 1
    cfg.simulation.base_seed = 880
    pad = len(str(cfg.simulation.num_steps))

    out_a = tmp_path / f"a_{cfg_name.replace('.yaml', '')}"
    out_b = tmp_path / f"b_{cfg_name.replace('.yaml', '')}"
    run_once(0, cfg, out_dir=out_a)
    run_once(0, cfg, out_dir=out_b)

    _assert_runs_identical(out_a, out_b, pad=pad)
