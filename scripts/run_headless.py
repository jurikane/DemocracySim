import random
from pathlib import Path
from datetime import datetime
import yaml
import numpy as np
from tqdm import tqdm

from src.config.loader import load_config, get_project_root
from src.model_setup import build_model_kwargs, make_model
from src.replay.replay_logger import ReplayLogger


def _snapshot(model):
    """Fast HxW uint8 color snapshot using model.color_cells list (row-major)."""
    return np.fromiter(
        (cell.color for cell in model.color_cells),
        dtype=np.uint8,
        count=model.height * model.width,
    ).reshape(model.height, model.width)


def _filter_model_kwargs(model_cfg: dict):
    """Filter model_cfg keys to match ParticipationModel.__init__ signature."""
    # Accept either dict-like or pydantic model
    if not isinstance(model_cfg, dict):
        try:
            model_cfg = dict(model_cfg)
        except Exception:
            # fallback: attempt to use .model attr
            model_cfg = getattr(model_cfg, "model", {})
    # If it's still not a dict, return empty
    if not isinstance(model_cfg, dict):
        return {}
    return model_cfg


def _cfg_get(cfg, key, default=None):
    """Access key from dict-like or attribute from pydantic model."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def run_once(run_id: int, model_cfg, sim_cfg, out_dir: Path):
    # sim_cfg may be dict or pydantic model
    base_seed_val = _cfg_get(sim_cfg, "base_seed", None)
    if base_seed_val is None:
        base_seed_val = random.SystemRandom().randint(0, 2 ** 31 - 1)
    base_seed = int(base_seed_val)
    run_seed = base_seed + int(run_id)
    random.seed(run_seed)
    np.random.seed(run_seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    rl = ReplayLogger(out_dir=out_dir, run_id=run_id, store_grid=bool(_cfg_get(sim_cfg, "store_grid", True)))

    # If model_cfg is a full AppConfig (has .model) prefer make_model
    try:
        if hasattr(model_cfg, "model"):
            model = make_model(model_cfg)
        else:
            model = None
    except Exception:
        model = None

    if model is None:
        # fallback to building kwargs manually
        try:
            kwargs = build_model_kwargs(model_cfg)
            from src.models.participation_model import ParticipationModel
            model = ParticipationModel(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate model: {e}")
    try:
        model.random.seed(run_seed)
    except Exception:
        pass
    rl.write_static(model)
    n_steps = int(_cfg_get(sim_cfg, "num_steps", 100))
    grid_interval = max(1, int(_cfg_get(sim_cfg, "grid_interval", 1)))
    for step in tqdm(range(n_steps), desc=f"run {run_id}"):
        model.step()
        grid_snapshot = None
        if _cfg_get(sim_cfg, "store_grid", True) and (step % grid_interval == 0):
            grid_snapshot = _snapshot(model)
        rl.append_step(step=step, model=model, grid_snapshot=grid_snapshot)
    rl.flush()
    # Write full AppConfig for robust replay
    rl.write_meta(config=model_cfg if hasattr(model_cfg, "model_dump") else model_cfg, seed=run_seed)


def _resolve_output_base_dir(conf) -> Path:
    """Return the base output folder for runs.

    Rules:
      - default: <project_root>/data/simulation_output
      - if conf.output.directory is absolute: use as-is
      - if conf.output.directory is relative: interpret relative to project root
    """
    project_root = get_project_root()
    default_dir = project_root / "data" / "simulation_output"

    output_cfg = getattr(conf, "output", None)
    if output_cfg is None:
        return default_dir

    configured = getattr(output_cfg, "directory", None)
    if configured is None:
        return default_dir

    candidate = Path(configured)
    if candidate.is_absolute():
        return candidate
    return project_root / candidate


def batch_run(config_file: str = None):
    conf = load_config(config_file)
    # conf is a pydantic AppConfig object
    model_cfg = conf.model
    sim_cfg = conf.simulation
    output_cfg = getattr(conf, "output", None)
    viz_cfg = conf.visualization
    # Determine base directory
    base_dir = _resolve_output_base_dir(conf)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = base_dir / ts
    run_root.mkdir(parents=True, exist_ok=True)
    with open(run_root / "config_used.yaml", "w") as f:
        # pydantic AppConfig -> dict (support v1/v2)
        if hasattr(conf, "model_dump"):
            yaml.safe_dump(conf.model_dump(), f)
        elif hasattr(conf, "dict"):
            yaml.safe_dump(conf.dict(), f)
        else:
            yaml.safe_dump(conf, f)

    # Ensure base_seed exists on sim_cfg for run_once
    base_seed = _cfg_get(sim_cfg, "base_seed", None)
    if base_seed is None:
        base_seed = random.SystemRandom().randint(0, 2 ** 31 - 1)
        try:
            if isinstance(sim_cfg, dict):
                sim_cfg["base_seed"] = int(base_seed)
            else:
                setattr(sim_cfg, "base_seed", int(base_seed))
        except Exception:
            pass
    runs = int(_cfg_get(sim_cfg, "runs", 1))
    for run_id in range(runs):
        this_out = run_root / f"run_{run_id}"
        this_out.mkdir(parents=True, exist_ok=True)
        # pass full conf (AppConfig) so run_once can call make_model if desired
        run_once(run_id, conf, sim_cfg, out_dir=this_out)
    print("All runs finished. Results saved to:", run_root)


if __name__ == "__main__":
    batch_run()
