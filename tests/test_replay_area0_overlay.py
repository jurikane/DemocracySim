from __future__ import annotations

import yaml

from src.config.schema import AppConfig
from src.replay.replay_server import ReplayModel
from src.viz.visualisation_elements import AreaStats


def test_replay_area_stats_includes_area_0(tmp_path):
    """Regression test: replay must not drop area_id=0 from area overlay inputs.

    The historical bug was in AreaStats.render() using `unique()[1:]`, which
    skipped the first area id (often 0).

    We test the actual data interface the overlay consumes:
    - ReplayModel must provide agent vars including AgentID==0
    - AreaStats must render a non-empty image for step > 0
    """
    run_dir = tmp_path / "run_0"
    run_dir.mkdir(parents=True)

    # Use an existing sample run from repo data.
    import shutil
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    sample_dir = repo_root / "data" / "simulation_output" / "20260112_144710" / "run_1"

    for name in ["static.json", "personalities.json", "meta.yaml"]:
        p = sample_dir / name
        if p.exists():
            shutil.copy(p, run_dir / name)

    (run_dir / "steps").mkdir()
    for step_name in ["step_0000.json", "step_0001.json"]:
        shutil.copy(sample_dir / "steps" / step_name, run_dir / "steps" / step_name)

    meta = yaml.safe_load((run_dir / "meta.yaml").read_text())
    appcfg = AppConfig.model_validate(meta["config"])

    model = ReplayModel(appcfg=appcfg, run_dir=run_dir)

    # Step once so AreaStats activates (it exits early at step==0)
    model.step()

    df = model.datacollector.get_agent_vars_dataframe()
    assert len(df) > 0
    assert 0 in df.index.get_level_values(1)

    html = AreaStats().render(model)
    assert isinstance(html, str)
    assert html != ""
