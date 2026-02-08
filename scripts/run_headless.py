import random
from pathlib import Path
from datetime import datetime
import yaml
from tqdm import tqdm
import argparse
import numpy as np

from src.config.loader import load_config, get_project_root
from src.model_setup import make_model
from src.utils.metrics import get_grid_colors
from src.logging.run_logger import RunLoggerV2


def run_once(run_id: int, cfg, out_dir: Path):
    """
    Execute a single simulation run and log results.
    Attributes:
        run_id (int): Identifier for the run (used for seeding).
        cfg (AppConfig): Full configuration object.
        out_dir (Path): Output directory for logging.
    """
    try:
        # Copy the config for each run to avoid mutation issues
        cfg_for_run = cfg.model_copy(deep=True)
        sim_cfg = cfg_for_run.simulation
        base_seed = int(sim_cfg.base_seed) or None
        if base_seed is None:
            base_seed = random.SystemRandom().randint(0, 2 ** 31 - 1)
            print(f"No base_seed in config; using random seed {base_seed}")
        run_seed = base_seed + int(run_id)
        store_grid = bool(getattr(sim_cfg, "store_grid", True))
        model_cfg_for_run = cfg_for_run.model
        model_cfg_for_run.seed = run_seed
        out_dir.mkdir(parents=True, exist_ok=True)
        n_steps = int(getattr(sim_cfg, "num_steps", 100))

        # Create model instance
        model = make_model(model_cfg_for_run)

        # Schema v2 logger (v2-only)
        rule_idx = int(getattr(model_cfg_for_run, "rule_idx", 0) or 0)
        v2 = RunLoggerV2(out_dir=out_dir, run_seed=run_seed, rule_idx=rule_idx, num_steps=n_steps, store_grid=store_grid)
        v2.write_static(model)
        v2.write_meta(cfg_for_run)
        v2.attach_to_model(model)

        # Write initial (pre-election) grid snapshot for UI convenience.
        # This is NOT part of schema v2 step indexing (parquet remains 1...N).
        if store_grid:
            (out_dir / "grids").mkdir(parents=True, exist_ok=True)
            pad = len(str(int(n_steps)))
            initial_grid = get_grid_colors(model)
            np.save(str(out_dir / "grids" / f"grid_{0:0{pad}d}.npy"), np.asarray(initial_grid))

    except Exception as e:
        raise RuntimeError(f"Failed to instantiate model: {e}")

    grid_interval = max(1, int(getattr(sim_cfg, "grid_interval", 1)))
    for step in tqdm(range(n_steps), desc=f"run {run_id}"):
        # Schema v2 uses 1-based step indexing. steps/area_steps color distributions
        # are captured pre-mutation; grid snapshots remain post-mutation.
        v2_step = step + 1
        v2.begin_step(v2_step)
        model.step()
        grid_snapshot = None
        if store_grid and (step % grid_interval == 0):
            grid_snapshot = get_grid_colors(model)
        # Schema v2 tables + grids (1-based)
        v2.log_step(step=v2_step, model=model, grid_snapshot=grid_snapshot)
        v2.end_step()

    v2.finalize()
    v2.detach_from_model(model)

    # NOTE: schema v2 meta.yaml has already been written above.


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
    """Run multiple simulation runs in batch mode based on the provided config.
    Attributes:
        config_file (str): Path to the YAML/TOML config file.
    """
    conf = load_config(config_file)
    # conf is a pydantic AppConfig object
    sim_cfg = conf.simulation
    # Ensure base_seed exists on sim_cfg for run_once
    base_seed = getattr(sim_cfg, "base_seed", None)
    if base_seed is None:
        base_seed = random.SystemRandom().randint(0, 2 ** 31 - 1)
        print(f"No base_seed specified in config; using random seed {base_seed}")
        setattr(sim_cfg, "base_seed", int(base_seed))
    # Determine base directory
    base_out_dir = _resolve_output_base_dir(conf)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = base_out_dir / ts
    run_root.mkdir(parents=True, exist_ok=True)
    # Save the used config for reference
    with open(run_root / "config_used.yaml", "w") as f:
        # pydantic AppConfig -> dict (support v1/v2)
        if hasattr(conf, "model_dump"):
            yaml.safe_dump(conf.model_dump(), f)
        else:
            # Fallback:
            yaml.safe_dump(conf, f)
    runs = int(getattr(sim_cfg, "runs", 1))
    for run_id in range(runs):
        this_out = run_root / f"run_{run_id}"
        this_out.mkdir(parents=True, exist_ok=True)
        # pass full conf (AppConfig) so run_once can call make_model
        run_once(run_id, conf, out_dir=this_out)
    print("All runs finished. Results saved to:", run_root)


def headless_main() -> None:
    """CLI entry point for headless batch run.
    Use: python -m scripts.run_headless --config config_file_name.yaml
    """
    parser = argparse.ArgumentParser(description="Run DemocracySim headless")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to YAML/TOML config (or name under configs/)")
    args = parser.parse_args()
    batch_run(config_file=args.config)


if __name__ == "__main__":
    headless_main()
