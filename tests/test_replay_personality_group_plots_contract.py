from __future__ import annotations

import pytest

from src.config.loader import load_config
from scripts.run_headless import run_once
from src.replay.replay_server import ReplayModel


@pytest.fixture()
def v2_run_dir(tmp_path):
    out_dir = tmp_path / "v2_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config("configs/test.yaml")
    cfg_for_run = cfg.model_copy(deep=True)
    setattr(cfg_for_run.simulation, "num_steps", 2)
    setattr(cfg_for_run.simulation, "store_grid", True)
    setattr(cfg_for_run.simulation, "grid_interval", 1)

    run_once(run_id=0, cfg=cfg_for_run, out_dir=out_dir)
    assert (out_dir / "static.json").exists()
    return out_dir


def test_replay_has_personality_group_info_for_plots(v2_run_dir):
    """Guardrail: replay v2 must expose personality group metadata for plots.

    This catches static.json key mismatches (e.g. writing personality_info instead
    of personality_group_info).
    """
    appcfg = load_config("configs/test.yaml")
    model = ReplayModel(appcfg=appcfg, run_dir=v2_run_dir)

    # Create-once plots are rendered at scheduler.steps == 0.
    assert getattr(model.scheduler, "steps", 0) == 0

    # Global personality group distribution plot needs these.
    assert getattr(model, "personality_groups", None) is not None
    assert getattr(model, "personality_groups").size > 0, "ReplayModel.personality_groups empty"

    pgd = getattr(model, "personality_group_distribution", None)
    assert pgd is not None, "ReplayModel.personality_group_distribution missing"
    assert len(pgd) == getattr(model, "personality_groups").shape[0]

    # Area stubs should exist and carry per-area distributions (even if zeros).
    assert getattr(model, "areas", None)
    for area in model.areas:
        dist = getattr(area, "personality_group_distribution", None)
        assert dist is not None, f"Area {getattr(area, 'unique_id', '?')} missing personality_group_distribution"
