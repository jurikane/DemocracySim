from __future__ import annotations

from scripts.run_headless import run_once
from src.config.loader import load_config
from src.replay.replay_server import ReplayModel
from src.viz.debug_viz import AreaDiagnosticsPanel
from tests.utils_meta import load_appcfg_from_meta


def test_replay_area_stats_includes_area_0(tmp_path):
    """Regression test: replay must not drop area_id=0 from area overlay inputs."""
    run_dir = tmp_path / "run_0"
    run_dir.mkdir(parents=True)

    conf = load_config('configs/toy.yaml')
    sim_cfg = conf.simulation

    sim_cfg = sim_cfg.model_copy(deep=True)
    sim_cfg.num_steps = 2
    sim_cfg.grid_interval = 1
    sim_cfg.store_grid = True

    run_once(0, conf, out_dir=run_dir)

    appcfg = load_appcfg_from_meta(run_dir)

    model = ReplayModel(appcfg=appcfg, run_dir=run_dir)

    model.step()  # activate AreaDiagnosticsPanel (it exits early at step==0)

    df = model.datacollector.get_agent_vars_dataframe()
    assert len(df) > 0
    assert 0 in df.index.get_level_values(1)

    html = AreaDiagnosticsPanel().render(model)
    assert isinstance(html, str)
    assert html != ""
