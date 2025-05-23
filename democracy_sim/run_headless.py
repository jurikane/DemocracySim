# fast_batch.py  ── headless, parallel & compact
import yaml, argparse, multiprocessing as mp, json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from participation_model import ParticipationModel
from tqdm import tqdm

# ─────────────────────────────── CONFIG ────────────────────────────────
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _snapshot(model):
    """
    Return an H×W uint8 array with the current grid colours.
    Uses the pre-allocated model.color_cells list, so no grid
    access is needed.
    """
    return np.fromiter(
        (cell.color for cell in model.color_cells),
        dtype=np.uint8,
        count=model.height * model.width
    ).reshape(model.height, model.width)


# ─────────────────────── SINGLE-RUN EXECUTION ─────────────────────────

def run_once(run_id, model_cfg, num_steps, store_grid, grid_interval, out_dir):
    np.random.seed(run_id)
    model = ParticipationModel(**model_cfg)

    # ——— dump static geometry ————————————————————————————
    # agents: (unique_id, x, y, personality_idx)
    agent_arr = np.array(
        [(a.unique_id, a.row, a.col, a.personality_idx)
         for a in model.voting_agents],
        dtype=np.int32
    )

    # areas: (unique_id, x, y, height, width)
    # note: Area stores its dimensions in _height/_width
    area_arr = np.array(
        [(ar.unique_id,
          ar.idx_field[0], ar.idx_field[1],
          ar._height, ar._width)
         for ar in model.areas],
        dtype=np.int32
    )

    np.savez_compressed(out_dir / f"static_{run_id}.npz",
                        agents=agent_arr,
                        areas=area_arr)

    # ----- grid data

    model_records = []
    grids = [] if store_grid else None

    for step in range(num_steps):
        model.step()

        row = model.datacollector.get_model_vars_dataframe().iloc[-1].to_dict()
        row["run_id"] = run_id
        row["step"]   = step
        model_records.append(row)

        if store_grid and step % grid_interval == 0:
            grids.append(_snapshot(model))

    df = pd.DataFrame(model_records)
    pq.write_table(pa.Table.from_pandas(df),
                   out_dir / f"model_{run_id}.parquet")

    if store_grid:
        np.savez_compressed(out_dir / f"grid_{run_id}.npz",
                            grid=np.stack(grids, axis=0))

    del model, model_records, grids
    return run_id


# ──────────────────────── PARALLEL BATCH DRIVER ───────────────────────

def _run_wrapper(args):
    """Top-level helper so it can be pickled by multiprocessing."""
    return run_once(*args)

def batch_run(cfg):
    sim_cfg   = cfg["simulation"]
    model_cfg = cfg["model"]

    num_runs      = sim_cfg.get("runs", 20)
    num_steps     = sim_cfg.get("num_steps", 1000)
    processes     = sim_cfg.get("processes", mp.cpu_count())
    store_grid    = sim_cfg.get("store_grid", False)
    grid_interval = max(1, sim_cfg.get("grid_interval", 1))

    out_dir = Path(cfg["output"].get("directory", "runs")) / \
              datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "meta.json", "w") as f:
        json.dump(dict(model=model_cfg, simulation=sim_cfg), f, indent=2)

    print(f"▶ Launching {num_runs} runs on {processes} processes ...")
    args = [(i, model_cfg, num_steps, store_grid, grid_interval, out_dir)
            for i in range(num_runs)]

    # --- no lambdas, only a top-level function ------------------------
    with mp.get_context("spawn").Pool(processes) as pool:
        for _ in tqdm(pool.imap_unordered(_run_wrapper, args), total=num_runs):
            pass

    print("✔ all runs finished → results in", out_dir)


# ──────────────────────────── CLI ENTRY ───────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fast parallel headless batch for Democracy-Sim"
    )
    parser.add_argument("--config", "-c", default="config/config.yaml",
                        help="YAML file with model/simulation/output blocks")
    config = load_yaml(parser.parse_args().config)
    batch_run(config)
