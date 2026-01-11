from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config.loader import load_config
from scripts.run_headless import run_once


def _load_step(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def test_headless_same_seed_produces_identical_first_steps(tmp_path: Path):
    """Critical determinism check for thesis runs.

    Contract: with same config + run_id (=> same run_seed), the first few
    replay artifacts should be byte-for-byte equivalent in meaning.

    We compare JSON content (parsed) and a couple of grid snapshots.
    """
    cfg = load_config("toy.yaml")

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    run_once(0, cfg, cfg.simulation, out_dir=out_a)
    run_once(0, cfg, cfg.simulation, out_dir=out_b)

    for rel in [
        Path("static.json"),
        Path("meta.yaml"),
        Path("steps/step_0000.json"),
        Path("steps/step_0001.json"),
    ]:
        assert (out_a / rel).exists(), rel
        assert (out_b / rel).exists(), rel

    s0_a = _load_step(out_a / "steps" / "step_0000.json")
    s0_b = _load_step(out_b / "steps" / "step_0000.json")
    assert s0_a == s0_b

    s1_a = _load_step(out_a / "steps" / "step_0001.json")
    s1_b = _load_step(out_b / "steps" / "step_0001.json")
    assert s1_a == s1_b

    # If grids are stored in this config, ensure identical arrays at step 0/1.
    g0_a = out_a / "grids" / "grid_0000.npy"
    g0_b = out_b / "grids" / "grid_0000.npy"
    if g0_a.exists() and g0_b.exists():
        np.testing.assert_array_equal(np.load(g0_a), np.load(g0_b))

    g1_a = out_a / "grids" / "grid_0001.npy"
    g1_b = out_b / "grids" / "grid_0001.npy"
    if g1_a.exists() and g1_b.exists():
        np.testing.assert_array_equal(np.load(g1_a), np.load(g1_b))

