from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_headless import run_once
from src.config.loader import load_config


@pytest.mark.phase1
def test_v2_headless_writes_no_legacy_step_json_or_static_v2(tmp_path: Path) -> None:
    """Guardrail: schema v2 hard-cut must not write legacy step-JSON artifacts."""
    cfg = load_config("configs/toy.yaml").model_copy(deep=True)
    cfg.simulation.num_steps = 2
    cfg.simulation.grid_interval = 1
    cfg.simulation.store_grid = True

    run_dir = tmp_path / "run_0"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_once(0, cfg, out_dir=run_dir)

    # Required v2 artifacts
    for name in ("meta.yaml", "static.json", "steps.parquet", "area_steps.parquet", "agents.parquet", "votes.parquet"):
        assert (run_dir / name).exists(), f"Missing required v2 artifact: {name}"

    # Required grid step 0 (pre-election)
    pad = len(str(int(cfg.simulation.num_steps)))
    assert (run_dir / "grids" / f"grid_{0:0{pad}d}.npy").exists()

    # Legacy artifacts must not exist
    assert not (run_dir / "static_v2.json").exists(), "static_v2.json should not be written in v2 hard-cut"
    assert not (run_dir / "steps").exists(), "legacy steps/ directory (step_*.json) should not be written"


@pytest.mark.phase1
def test_replay_refuses_non_v2_run_dir(tmp_path: Path) -> None:
    """Guardrail: replay must refuse run dirs without schema v2 meta.yaml."""
    from src.replay.replay_server import ReplayData

    # Empty dir: no meta.yaml
    run_dir = tmp_path / "not_a_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError):
        ReplayData(run_dir)
