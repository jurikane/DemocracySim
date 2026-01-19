import tempfile
import numpy as np
import json
import yaml
from pathlib import Path
from src.config.loader import load_config
from src.model_setup import make_model
from src.replay.replay_logger import ReplayLogger


def test_replay_logger_smoke():
    cfg = load_config('toy.yaml')
    model_cfg = cfg.model
    # instantiate model via make_model
    model = make_model(model_cfg)
    # run 2 steps and log
    td = Path(tempfile.mkdtemp())/"run_test"
    td.mkdir(parents=True, exist_ok=True)
    n = cfg.simulation.num_steps
    rl = ReplayLogger(out_dir=td, num_steps=n, run_id=0, store_grid=True)
    rl.write_static(model)
    for step in range(2):
        model.step()
        grid = np.fromiter((c.color for c in model.color_cells),
                           dtype=np.uint8,
                           count=model.height*model.width).reshape(
            model.height, model.width)

        rl.append_step(step, model, grid)
    rl.write_meta(config=cfg)

    # check files
    assert (td / 'static.json').exists()
    assert (td / 'meta.yaml').exists()
    pad = len(str(n))
    assert (td / "steps" / f"step_{0:0{pad}d}.json").exists()
    assert (td / "steps" / f"step_{1:0{pad}d}.json").exists()
    assert (td / "grids" / f"grid_{0:0{pad}d}.npy").exists()
    assert (td / "grids" / f"grid_{1:0{pad}d}.npy").exists()

    # step json should NOT contain heavyweight grid keys
    d0 = json.loads((td / "steps" / f"step_{0:0{pad}d}.json").read_text())
    assert 'step' in d0
    assert 'GridColors' not in d0

    # meta should contain format_version
    meta = yaml.safe_load((td / 'meta.yaml').read_text())
    assert meta.get('format_version') == 1
