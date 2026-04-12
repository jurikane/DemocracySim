import random
from pathlib import Path
from datetime import datetime
import yaml
import hashlib
from tqdm import tqdm
import argparse
import numpy as np

from src.config.loader import load_config, resolve_output_dir
from src.config.schema import AppConfig
from src.model_setup import make_model
from src.utils.metrics import get_grid_colors
from src.logging.run_logger import RunLogger


def run_once(run_id: int, cfg: AppConfig, out_dir: Path):
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
        base_seed = sim_cfg.base_seed
        if base_seed is None:
            base_seed = random.SystemRandom().randint(0, 2 ** 31 - 1)
            print(f"No base_seed in config; using random seed {base_seed}")
        run_seed = base_seed + int(run_id)
        store_grid = sim_cfg.store_grid
        # Even in headless DOE runs (store_grid=false), keep minimal grid snapshots
        # (step 1 + last step) so summary PDFs can show first/last grids.
        store_summary_grids = True
        model_cfg_for_run = cfg_for_run.model
        model_cfg_for_run.seed = run_seed
        out_dir.mkdir(parents=True, exist_ok=True)
        n_steps = sim_cfg.num_steps

        # Create model instance
        model = make_model(model_cfg_for_run, enable_datacollector=False)

        # Schema v3 logger
        rule_idx = model_cfg_for_run.rule_idx
        logger = RunLogger(
            out_dir=out_dir,
            run_seed=run_seed,
            rule_idx=rule_idx,
            num_steps=n_steps,
            store_grid=(store_grid or store_summary_grids),
        )
        logger.write_static(model)
        cfg_ref = Path("..") / "config_used.yaml"
        cfg_ref_path = (out_dir / cfg_ref).resolve()
        if not cfg_ref_path.exists():
            # For direct run_once usage (no batch_run), write a canonical config copy.
            with cfg_ref_path.open("w", encoding="utf-8") as f:
                cfg_dump = cfg.model_dump(mode="json")
                yaml.safe_dump(cfg_dump, f)
        cfg_hash = hashlib.sha256(cfg_ref_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
        logger.write_meta(cfg_for_run, config_ref=cfg_ref, config_hash=cfg_hash, model=model)
        logger.attach_to_model(model)

        # Write initial (pre-election) grid snapshot for UI convenience.
        # This is NOT part of schema step indexing (parquet remains 1...N).
        if store_grid:
            (out_dir / "grids").mkdir(parents=True, exist_ok=True)
            pad = len(str(int(n_steps)))
            initial_grid = get_grid_colors(model)
            np.save(str(out_dir / "grids" / f"grid_{0:0{pad}d}.npy"), np.asarray(initial_grid))

    except (RuntimeError, ValueError, TypeError, AttributeError, OSError, yaml.YAMLError) as e:
        raise RuntimeError(f"Failed to instantiate model: {e}")

    grid_interval = max(1, sim_cfg.grid_interval)
    for step in tqdm(range(n_steps), desc=f"run {run_id}"):
        # Schema uses 1-based step indexing. steps/area_steps color distributions
        # are captured pre-mutation; grid snapshots are also pre-mutation.
        recorded_step = step + 1
        logger.begin_step(recorded_step)
        model.step()
        grid_snapshot = None
        should_write_grid = store_grid and (step % grid_interval == 0)
        # Always provide first/last election-time snapshots for summary pages.
        if (not should_write_grid) and store_summary_grids and (step == 0 or step == n_steps - 1):
            should_write_grid = True
        if should_write_grid:
            grid_snapshot = get_grid_colors(model)
        logger.log_step(step=recorded_step, model=model, grid_snapshot=grid_snapshot)
        logger.end_step()

    logger.finalize()
    logger.detach_from_model(model)

    # NOTE: schema meta.yaml has already been written above.


def batch_run(config_file: str = None, out_root: str | None = None, num_steps: int | None = None):
    """Run multiple simulation runs in batch mode based on the provided config.
    Attributes:
        config_file (str): Path to the YAML/TOML config file.
    """
    conf = load_config(config_file)
    # conf is a pydantic AppConfig object
    sim_cfg = conf.simulation
    # Ensure base_seed exists on sim_cfg for run_once
    base_seed = sim_cfg.base_seed
    if base_seed is None:
        base_seed = random.SystemRandom().randint(0, 2 ** 31 - 1)
        print(f"No base_seed specified in config; using random seed {base_seed}")
        sim_cfg.base_seed = base_seed
    if num_steps is not None:
        sim_cfg.num_steps = int(num_steps)
    # Determine output directory (explicit path wins; otherwise timestamp under config output dir)
    if out_root:
        run_root = Path(out_root)
    else:
        base_out_dir = resolve_output_dir(conf)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = base_out_dir / ts
    run_root.mkdir(parents=True, exist_ok=True)
    # Save the used config for reference
    with open(run_root / "config_used.yaml", "w", encoding="utf-8") as f:
        cfg_dump = conf.model_dump(mode="json")
        yaml.safe_dump(cfg_dump, f)
    runs = sim_cfg.runs
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
    parser.add_argument("--out-root", type=str, default=None,
                        help="Explicit output root directory (no timestamp suffix added)")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Override simulation.num_steps for this run")
    args = parser.parse_args()
    batch_run(config_file=args.config, out_root=args.out_root, num_steps=args.num_steps)


if __name__ == "__main__":
    headless_main()
