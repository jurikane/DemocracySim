from __future__ import annotations

from pathlib import Path

from src.config.loader import load_config
from scripts.run_headless import run_once
from src.replay.replay_server import ReplayData


def test_replay_can_iterate_all_steps(tmp_path: Path):
    """Replay contract check: produced run dir can be fully traversed.

    We don't assert any model logic here; we only verify the on-disk
    representation is internally consistent and complete.
    """
    cfg = load_config("toy.yaml")
    out_dir = tmp_path / "run"

    run_once(0, cfg, cfg.simulation, out_dir=out_dir)

    data = ReplayData(out_dir)
    assert len(data) > 0

    # Ensure step files are present and readable
    last_step = -1
    for i in range(len(data)):
        rec = data.load_step(i)
        step = int(rec.get("step", i))
        assert step >= 0
        assert step > last_step
        last_step = step
        # If a grid exists for this step, it must be loadable.
        grid = data.load_grid(step)
        if grid is not None:
            assert grid.shape == (cfg.model.height, cfg.model.width)

    assert last_step >= 0

