from __future__ import annotations

import pytest

from src.config.loader import load_config
from scripts.run_headless import run_once
from src.replay.replay_server import ReplayModel
from src.viz.visualization_elements import MainMetricsElement


@pytest.fixture()
def v2_run_dir(tmp_path):
    out_dir = tmp_path / "v2_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config("configs/toy.yaml")
    cfg_for_run = cfg.model_copy(deep=True)
    setattr(cfg_for_run.simulation, "num_steps", 2)
    setattr(cfg_for_run.simulation, "store_grid", True)
    setattr(cfg_for_run.simulation, "grid_interval", 1)

    run_once(run_id=0, cfg=cfg_for_run, out_dir=out_dir)
    assert (out_dir / "static.json").exists()
    return out_dir


def test_replay_has_personality_group_info_for_plots(v2_run_dir):
    """Guardrail: replay v2 must expose personality (preference) group metadata for plots.

    This catches static.json key mismatches (e.g. writing personality_info instead
    of personality_group_info).
    """
    appcfg = load_config("configs/toy.yaml")
    model = ReplayModel(appcfg=appcfg, run_dir=v2_run_dir)

    # Create-once plots are rendered at scheduler.steps == 0.
    assert model.scheduler.steps == 0

    # Global personality (preference) group distribution plot needs these.
    assert model.personality_groups.size > 0, "ReplayModel.personality_groups empty"

    pgd = model.personality_group_distribution
    assert pgd is not None, "ReplayModel.personality_group_distribution missing"
    assert len(pgd) == model.personality_groups.shape[0]

    # Area stubs should exist and carry per-area distributions (even if zeros).
    assert model.areas
    for area in model.areas:
        dist = area.personality_group_distribution
        assert dist is not None, f"Area {area.unique_id} missing personality_group_distribution"


def test_replay_main_metrics_panel_renders_after_first_recorded_step(v2_run_dir) -> None:
    appcfg = load_config("configs/toy.yaml")
    model = ReplayModel(appcfg=appcfg, run_dir=v2_run_dir)

    model.step()
    html = MainMetricsElement().render(model)

    assert "Turnout" in html
    assert "Inequality" in html
    assert "Outcome quality" in html
