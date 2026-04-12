import tempfile
import yaml
from pathlib import Path
import pandas as pd

from src.config.loader import load_config
from src.model_setup import make_model
from src.logging.run_logger import RunLogger
from src.utils.metrics import get_grid_colors


def test_run_logger_smoke():
    cfg = load_config('toy.yaml')
    model_cfg = cfg.model
    # instantiate model via make_model
    model = make_model(model_cfg)
    # run 2 steps and log
    td = Path(tempfile.mkdtemp()) / "run_test"
    td.mkdir(parents=True, exist_ok=True)

    n = int(cfg.simulation.num_steps)
    run_seed = int(cfg.simulation.base_seed)
    rule_idx = int(cfg.model.rule_idx)

    logger = RunLogger(out_dir=td, num_steps=n, run_seed=run_seed, rule_idx=rule_idx, store_grid=True)
    logger.write_static(model)
    logger.write_meta(cfg)
    logger.attach_to_model(model)

    # run 2 steps and log
    for step in range(1, 3):
        logger.begin_step(step)
        model.step()
        grid = get_grid_colors(model)
        logger.log_step(step=step, model=model, grid_snapshot=grid)
        logger.end_step()

    logger.finalize()
    logger.detach_from_model(model)

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
    assert meta.get('schema', {}).get('name') == 'output_schema_v3'
