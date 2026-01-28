import tempfile
import yaml
from pathlib import Path
import pandas as pd

from src.config.loader import load_config
from src.model_setup import make_model
from src.logging.run_logger_v2 import RunLoggerV2
from src.utils.metrics import get_grid_colors


def test_run_logger_v2_smoke():
    cfg = load_config('toy.yaml')
    model_cfg = cfg.model
    # instantiate model via make_model
    model = make_model(model_cfg)
    # run 2 steps and log
    td = Path(tempfile.mkdtemp()) / "run_test"
    td.mkdir(parents=True, exist_ok=True)

    n = int(cfg.simulation.num_steps)
    run_seed = int(getattr(cfg.simulation, 'base_seed', 42) or 42)
    rule_idx = int(getattr(cfg.model, 'rule_idx', 0) or 0)

    v2 = RunLoggerV2(out_dir=td, num_steps=n, run_seed=run_seed, rule_idx=rule_idx, store_grid=True)
    v2.write_static(model)
    v2.write_meta(cfg)
    v2.attach_to_model(model)

    # run 2 steps and log
    for step in range(1, 3):
        v2.begin_step(step)
        model.step()
        grid = get_grid_colors(model)
        v2.log_step(step=step, model=model, grid_snapshot=grid)
        v2.end_step()

    v2.finalize()
    v2.detach_from_model(model)

    # check files
    assert (td / 'static.json').exists()
    assert (td / 'meta.yaml').exists()
    assert (td / 'steps.parquet').exists()
    assert (td / 'area_steps.parquet').exists()
    assert (td / 'agents.parquet').exists()
    assert (td / 'votes.parquet').exists()

    pad = len(str(n))
    assert (td / 'grids' / f'grid_{1:0{pad}d}.npy').exists()
    assert (td / 'grids' / f'grid_{2:0{pad}d}.npy').exists()

    # basic sanity: first and second step rows exist
    steps = pd.read_parquet(td / 'steps.parquet').sort_values('step')
    assert (steps['step'].astype(int) == [1, 2]).all()

    meta = yaml.safe_load((td / 'meta.yaml').read_text())
    assert meta.get('schema', {}).get('name') == 'output_schema_v2'
