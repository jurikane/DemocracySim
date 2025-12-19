"""Replay script: given a run directory produced by run_headless,
rebuild the model from the stored config/seed (if available) and apply
saved grid snapshots to the model so callers can inspect or visualize them.

Usage: python -m scripts.run_replay <run_dir>
"""
import sys
from pathlib import Path
import yaml
import json
import numpy as np

from src.config.schema import AppConfig
from src.model_setup import make_model
from src.replay.replay_server import make_replay_server


def _apply_grid_to_model(model, arr: np.ndarray):
    """Apply a HxW numpy array of color indices to model.color_cells in
    the same row-major order used by the snapshot helper.
    """
    flat = arr.ravel()
    cells = getattr(model, "color_cells", None)
    if cells is None:
        return
    for i, cell in enumerate(cells):
        try:
            cell.color = int(flat[i])
        except Exception:
            # best-effort
            pass


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.run_replay <run_dir>")
        return
    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print("Run directory does not exist:", run_dir)
        return

    meta_path = run_dir / "meta.yaml"
    static_path = run_dir / "static.json"
    meta = None
    if meta_path.exists():
        meta = yaml.safe_load(meta_path.read_text())
    static = None
    if static_path.exists():
        static = json.loads(static_path.read_text())

    model = None
    if meta is not None and "config" in meta:
        cfg_raw = meta["config"]
        try:
            appcfg = AppConfig.model_validate(cfg_raw)
        except Exception:
            appcfg = None
        if appcfg is not None:
            try:
                model = make_model(appcfg)
                print("Rebuilt model from meta config.")
            except Exception as e:
                print("Failed to rebuild model from config:", e)
    if model is None and static is not None:
        # Minimal fallback: instantiate a tiny model using static fields
        try:
            # create a minimal AppConfig-like dict
            model_stub = type("Stub", (), {})()
            # reuse model_setup.make_model expects AppConfig; skip and construct smaller
            from src.models.participation_model import ParticipationModel
            from src.model_setup import build_model_kwargs
            # Build kwargs from static where possible
            kwargs = {"height": static.get("height"), "width": static.get("width"),
                      "num_agents": static.get("num_agents"),
                      "num_colors": static.get("num_colors"), "num_personalities": 1,
                      "mu": 0.01, "election_impact_on_mutation": 1.0, "common_assets": 100,
                      "known_cells": 1, "num_areas": static.get("num_areas", 1),
                      "av_area_height": 1, "av_area_width": 1, "area_size_variance": 0.0,
                      "patch_power": 1.0, "color_patches_steps": 1, "heterogeneity": 0.1,
                      "rule_idx": 0, "distance_idx": 0, "election_costs": 1, "max_reward": 1}
            model = ParticipationModel(**{k: v for k, v in kwargs.items() if v is not None})
            print("Built minimal model from static.json fallback.")
        except Exception as e:
            print("Failed to build fallback model:", e)

    steps_dir = run_dir / "steps"
    grids_dir = run_dir / "grids"
    step_files = sorted(steps_dir.glob("step_*.json")) if steps_dir.exists() else []
    if not step_files:
        print("No step files found in:", steps_dir)
        return

    for sf in step_files:
        data = json.loads(sf.read_text())
        step = data.get("step")
        print(f"Step {step}: reporters={list(k for k in data.keys() if k != 'step')}")
        grid_file = grids_dir / f"grid_{int(step):04d}.npy"
        if grid_file.exists():
            arr = np.load(str(grid_file))
            print(f"  grid shape: {arr.shape}, min/max: {arr.min()}/{arr.max()}")
            if model is not None:
                _apply_grid_to_model(model, arr)
                # Optionally, collect datacollector after applying
                try:
                    if hasattr(model, "datacollector") and model.datacollector is not None:
                        df = model.datacollector.get_model_vars_dataframe()
                        if len(df) > 0:
                            last = df.iloc[-1].to_dict()
                            print("  model reporters (last):", {k: last[k] for k in last})
                except Exception:
                    pass

    # After CLI replay summary, offer to start a browser replay server
    if meta is not None and "config" in meta:
        try:
            appcfg = AppConfig.model_validate(meta["config"])
            server = make_replay_server(appcfg, run_dir)
            print("Starting replay server on http://127.0.0.1:8521 ...")
            server.launch(open_browser=True)
        except Exception as e:
            print("Could not start replay server:", e)

if __name__ == '__main__':
    main()
