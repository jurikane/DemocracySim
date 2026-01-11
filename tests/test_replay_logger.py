import tempfile
from pathlib import Path
from src.config.loader import load_config
from src.model_setup import make_model
from src.replay.replay_logger import ReplayLogger


def test_replay_logger_smoke():
    cfg = load_config('toy.yaml')
    model_cfg = cfg.model
    # instantiate model via make_model
    model = make_model(cfg)
    # run 2 steps and log
    td = Path(tempfile.mkdtemp())/"run_test"
    td.mkdir(parents=True, exist_ok=True)
    rl = ReplayLogger(out_dir=td, run_id=0, store_grid=True)
    rl.write_static(model)
    for step in range(2):
        model.step()
        grid = None
        try:
            import numpy as np
            grid = np.fromiter((c.color for c in model.color_cells), dtype=np.uint8, count=model.height*model.width).reshape(model.height, model.width)
        except Exception:
            grid = None
        rl.append_step(step, model, grid)
    rl.write_meta({'model': model_cfg, 'simulation': cfg.simulation}, seed=123)

    # check files
    assert (td / 'static.json').exists()
    assert (td / 'meta.yaml').exists()
    assert (td / 'steps' / 'step_0000.json').exists()
    assert (td / 'steps' / 'step_0001.json').exists()
    assert (td / 'grids' / 'grid_0000.npy').exists()
    assert (td / 'grids' / 'grid_0001.npy').exists()

    # step json should NOT contain heavyweight grid keys
    import json
    d0 = json.loads((td / 'steps' / 'step_0000.json').read_text())
    assert 'step' in d0
    assert 'GridColors' not in d0

    # meta should contain format_version
    import yaml
    meta = yaml.safe_load((td / 'meta.yaml').read_text())
    assert meta.get('format_version') == 1
