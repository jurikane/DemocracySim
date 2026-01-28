from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.loader import load_config
from scripts.run_headless import run_once


def _load_step(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def test_headless_same_seed_produces_identical_first_steps(tmp_path: Path):
    """Critical determinism check for thesis runs.

    Contract (schema v2): with same config + run_id (=> same run_seed), the first few
    logged artifacts should be identical in meaning.

    We compare parquet content (rows) and a couple of grid snapshots.
    """
    cfg = load_config("toy.yaml")
    pad = len(str(cfg.simulation.num_steps))

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    run_once(0, cfg, out_dir=out_a)
    run_once(0, cfg, out_dir=out_b)

    for rel in [
        Path("static.json"),
        Path("meta.yaml"),
        Path("steps.parquet"),
        Path("area_steps.parquet"),
        Path("agents.parquet"),
    ]:
        assert (out_a / rel).exists(), rel
        assert (out_b / rel).exists(), rel

    # Compare first two step rows in steps.parquet
    s_a = pd.read_parquet(out_a / "steps.parquet").sort_values("step").reset_index(drop=True)
    s_b = pd.read_parquet(out_b / "steps.parquet").sort_values("step").reset_index(drop=True)
    pd.testing.assert_frame_equal(s_a.head(2), s_b.head(2), check_dtype=False)

    # If grids are stored in this config, ensure identical arrays at step 1/2.
    g1_a = out_a / "grids" / f"grid_{1:0{pad}d}.npy"
    g1_b = out_b / "grids" / f"grid_{1:0{pad}d}.npy"
    if g1_a.exists() and g1_b.exists():
        np.testing.assert_array_equal(np.load(g1_a), np.load(g1_b))

    g2_a = out_a / "grids" / f"grid_{2:0{pad}d}.npy"
    g2_b = out_b / "grids" / f"grid_{2:0{pad}d}.npy"
    if g2_a.exists() and g2_b.exists():
        np.testing.assert_array_equal(np.load(g2_a), np.load(g2_b))
