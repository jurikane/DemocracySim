"""
ReplayLogger used by scripts/run_headless.py to store a simulation
run for later replay.

Storage layout (under out_dir):
- static.json         -- static model info
- meta.yaml           -- config and seed used
- steps/step_0000.json -- per-step scalar info (datacollector values etc.)
- grids/grid_0000.npy -- optional grid snapshot (numpy array) per stored step

API:
- ReplayLogger(out_dir: Path, num_steps: int, run_id: int, store_grid: bool=True)
- write_static(model)
- append_step(step: int, model, grid_snapshot: np.ndarray | None)
- flush()
- write_meta(config: dict)
"""
from __future__ import annotations

import json
import yaml
from pathlib import Path
from typing import Optional, Any
import numpy as np
from pandas import Series, Timestamp
import os
from src.utils.metrics import (get_area_border_grid, get_agents_per_cell_grid,
                               gini_index_0_100, get_agent_strings_per_cell_grid)


class ReplayLogger:
    def __init__(self, out_dir: Path, num_steps: int, run_id: int = 0,
                 store_grid: bool = True):
        self.out_dir = Path(out_dir)
        self.run_id = int(run_id)
        self.store_grid = bool(store_grid)
        self.steps_dir = self.out_dir / "steps"
        self.grids_dir = self.out_dir / "grids"
        self.num_steps = num_steps
        self.pad = len(str(num_steps)) or 4
        os.makedirs(self.steps_dir, exist_ok=True)
        if self.store_grid:
            os.makedirs(self.grids_dir, exist_ok=True)

    def _step_filename(self, step: int) -> Path:
        return self.steps_dir / f"step_{step:0{self.pad}d}.json"

    def _grid_filename(self, step: int) -> Path:
        return self.grids_dir / f"grid_{step:0{self.pad}d}.npy"

    def write_static(self, model: Any) -> None:
        """Write static information about the model to disk.

        We intentionally write a flattened/serializable summary rather than
        pickling the whole model.
        """
        # Normalize personalities for JSON (ReplayServer expects a list/array)
        raw_personalities = getattr(model, "personalities", None)
        if raw_personalities is None:
            personalities: list[Any] = []
        else:
            # if model.personalities is a numpy array; ensure plain python types
            personalities = _to_python(np.asarray(raw_personalities))

        static = {
            "format_version": 1,
            "height": int(getattr(model, "height", None)),
            "width": int(getattr(model, "width", None)),
            "total_voters": int(len(getattr(model, "voting_agents", []) or [])),
            "num_agents": int(getattr(model, "num_agents", None)),
            "num_colors": int(getattr(model, "num_colors", None)),
            "num_areas": int(getattr(model, "num_areas", None)),
            "num_voters_per_area": {},
            "voter_personalities": {},
            "personality_info": {
                "global_distribution": [],
                "personalities": personalities,
                "areas": {},
            },
            "voter_positions": {},
            "step_indexing": {
                "meaning": "post_step",
                "first_recorded_step": 0,
                "step_file": f"step_%0{self.pad}d.json",
                "grid_file": f"grid_%0{self.pad}d.npy",
            },
            "artifacts": {
                "area_borders": "area_borders.npy",
                "agents_per_cell": "agents_per_cell.npy",
            },
        }

        # --- Personality distributions (areas + global) ---
        # Count personalities globally via integer personality_idx (fast)
        voters = list(getattr(model, "voting_agents", []) or [])
        if len(personalities) > 0 and len(voters) > 0:
            p_idx = np.fromiter((int(v.personality_idx) for v in voters), dtype=np.int64, count=len(voters))
            counts = np.bincount(p_idx, minlength=len(personalities))
            denom = counts.sum()
            static["personality_info"]["global_distribution"] = (counts / denom).tolist() if denom > 0 else [0.0] * len(personalities)
        else:
            static["personality_info"]["global_distribution"] = [0.0] * len(personalities)

        # Save static voter numbers per area + per-area personality distributions
        for a in getattr(model, "areas", []) or []:
            aid = str(a.unique_id)
            n_agents = int(getattr(a, "num_agents", 0) or 0)
            static["num_voters_per_area"][aid] = n_agents

            # Prefer precomputed distribution if present; otherwise compute from agents
            dist = getattr(a, "personality_distribution", None)
            if dist is None:
                agents = list(getattr(a, "agents", []) or [])
                if len(personalities) > 0 and len(agents) > 0:
                    a_idx = np.fromiter((int(ag.personality_idx) for ag in agents), dtype=np.int64, count=len(agents))
                    a_counts = np.bincount(a_idx, minlength=len(personalities))
                    denom = a_counts.sum()
                    dist = (a_counts / denom).tolist() if denom > 0 else [0.0] * len(personalities)
                else:
                    dist = [0.0] * len(personalities)
            else:
                # Ensure JSON-serializable python list
                dist = _to_python(dist)

            static["personality_info"]["areas"][aid] = {
                "num_agents": n_agents,
                "personality_distribution": dist,
            }

        # Save voter personalities (ids -> personality_idx)
        for voter in voters:
            v_id = str(voter.unique_id)
            static["voter_personalities"][v_id] = int(voter.personality_idx)
            # static["voter_positions"][v_id] = str(voter.position)

        with open(self.out_dir / "static.json", "w") as f:
            json.dump(static, f, indent=2)

        # Store area border grid (HxW bool) as .npy for fast load
        borders = get_area_border_grid(model)
        border_file = self.out_dir / "area_borders.npy"
        np.save(str(border_file), np.asarray(borders, dtype=bool))

        # Store static agents-per-cell counts (HxW int)
        apc = get_agents_per_cell_grid(model)
        agents_per_cell_file = self.out_dir / "agents_per_cell.npy"
        np.save(str(agents_per_cell_file), np.asarray(apc, dtype=np.int32))

        # Store static agent strings per cell (HxW str)
        aspc = get_agent_strings_per_cell_grid(model)
        agent_strs_file = self.out_dir / "agent_strings_per_cell.npy"
        np.save(str(agent_strs_file), np.asarray(aspc, dtype=str))


    def append_step(self, step: int, model: Any, grid_snapshot: Optional[np.ndarray] = None) -> None:
        """Append per-step data.

        Contract (Schema v1):
        - steps/step_XXXX.json stores per-step model vars + area vars in a replay-friendly shape
        - grids/grid_XXXX.npy (optional) stores HxW int array of colors

        IMPORTANT:
        - We intentionally do not store large arrays in JSON (e.g. GridColors).
        - Step indexing convention is defined in static.json.
        """
        step_data: dict = {
            "format_version": 1,
            "step": int(step),
            "model": {},
            "areas": {},
        }

        # --- Global model scalars (for charts) ---
        if hasattr(model, "datacollector") and model.datacollector is not None:
            mrep = model.datacollector.get_model_vars_dataframe()
            if len(mrep):
                last = mrep.iloc[-1].to_dict()
                # Drop heavy / redundant fields
                for heavy_key in ("GridColors",):
                    last.pop(heavy_key, None)
                step_data["model"].update({k: _to_python(v) for k, v in last.items()})
        # try:
        if model.areas is not None:
            areas = model.areas
            for area in areas:
                area_id = int(getattr(area, "unique_id"))
                # Compute Gini index from the areas agents assets
                assets = [a.assets for a in getattr(area, "agents", [])]
                gini_area = gini_index_0_100(assets)

                step_data["areas"][str(area_id)] = {
                    "VoterTurnout": _to_python(getattr(area, "voter_turnout", None)),
                    "DistToReality": _to_python(getattr(area, "dist_to_reality", None)),
                    "ColorDistribution": _to_python(getattr(area, "color_distribution", None)),
                    "ElectionResults": _to_python(getattr(area, "voted_ordering", None)),
                    "GiniIndex": _to_python(gini_area),
                }
        # except Exception:
        #     # Fallback to Datacollector (slower):
        #     valid_area_ids = [area.unique_id for area in model.areas]
        #     # allow optional global area id
        #     valid_area_ids.append(-1)
        #     try:
        #         if hasattr(model, "datacollector") and model.datacollector is not None:
        #             adf = model.datacollector.get_agent_vars_dataframe()
        #             if adf is not None and len(adf) > 0:
        #                 try:
        #                     last_step = adf.index.get_level_values(0).max()
        #                     adf_step = adf.xs(last_step, level=0)
        #                 except Exception:
        #                     adf_step = adf
        #
        #                 for aid, row in adf_step.iterrows():
        #                     if aid not in valid_area_ids:
        #                         continue
        #                     step_data["areas"][str(aid)] = {
        #                         "VoterTurnout": _to_python(row.get("VoterTurnout")),
        #                         "DistToReality": _to_python(row.get("DistToReality")),
        #                         "ColorDistribution": _to_python(row.get("ColorDistribution")),
        #                         "ElectionResults": _to_python(row.get("ElectionResults")),
        #                         "GiniIndex": _to_python(row.get("GiniIndex")),
        #                     }
        #     except Exception:
        #         pass

        step_file = self._step_filename(step)
        with open(step_file, "w") as f:
            json.dump(step_data, f, indent=2)

        if self.store_grid and grid_snapshot is not None:
            grid_file = self._grid_filename(step)
            arr = np.asarray(grid_snapshot)
            np.save(str(grid_file), arr)

    def flush(self) -> None:
        """Provided for API compatibility.
        """
        return

    def write_meta(self, config) -> None:
        """
        Write meta information about each run.
        Args:
        - config: AppConfig
        """
        #
        meta = {
            "format_version": 1,
            "schema": {
                "name": "replay_schema_v1",
                "step_indexing": "post_step",
            },
            "config": {
                "model": _to_serializable(config.model) or None,
                "simulation": _to_serializable(config.simulation) or None,
                "visualization": _to_serializable(config.visualization) or None,
            },
        }
        with open(self.out_dir / "meta.yaml", "w") as f:
            yaml.safe_dump(meta, f)


def _to_python(obj: Any) -> Any:
    """Convert numpy/scalar types to plain python types for JSON serialization."""
    if obj is None:
        return None
    if isinstance(obj, (int, float, str, bool)):
        return obj
    # numpy scalars
    if isinstance(obj, (np.generic, np.ndarray)):
        if getattr(obj, "ndim", 0) == 0:
            return obj.item()
        return obj.tolist()
    # pandas types (Series, Timestamp etc.)
    if isinstance(obj, (Series, Timestamp)):
        return obj.to_json()

    # fallback
    return json.loads(json.dumps(obj, default=str))


def _to_serializable(obj: Any) -> Any:
    """Convert objects (including pydantic models) into serializable primitives."""
    # Prefer Pydantic v2 API first to avoid deprecation warnings
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        return _to_serializable(obj.model_dump())
    # Fallback to Pydantic v1 API if present
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        return _to_serializable(obj.dict())

    # dict
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    # simple types
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    # numpy/pandas
    return _to_python(obj)
