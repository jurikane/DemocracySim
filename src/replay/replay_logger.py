"""
Simple ReplayLogger used by scripts/run_headless.py to store a simulation
run for later replay.

Storage layout (under out_dir):
- static.json         -- static model info (grid size, num_agents, etc.)
- meta.yaml           -- config and seed used
- steps/step_0000.json -- per-step scalar info (datacollector values etc.)
- grids/grid_0000.npy -- optional grid snapshot (numpy array) per stored step

API:
- ReplayLogger(out_dir: Path, run_id: int, store_grid: bool=True)
- write_static(model)
- append_step(step: int, model, grid_snapshot: np.ndarray | None)
- flush()
- write_meta(config: dict, seed: int)

This implementation keeps files human-readable where possible and fast
(numpy .npy for arrays).
"""
from __future__ import annotations

import json
import yaml
from pathlib import Path
from typing import Optional, Any
import numpy as np
import os
from src.utils.metrics import get_area_border_grid, get_agents_per_cell_grid


class ReplayLogger:
    def __init__(self, out_dir: Path, run_id: int = 0, store_grid: bool = True):
        self.out_dir = Path(out_dir)
        self.run_id = int(run_id)
        self.store_grid = bool(store_grid)
        self.steps_dir = self.out_dir / "steps"
        self.grids_dir = self.out_dir / "grids"
        os.makedirs(self.steps_dir, exist_ok=True)
        if self.store_grid:
            os.makedirs(self.grids_dir, exist_ok=True)
        # Buffer to hold small per-step json objects before flush
        self._step_buffer: list[tuple[int, dict]] = []

    def _step_filename(self, step: int) -> Path:
        return self.steps_dir / f"step_{step:04d}.json"

    def _grid_filename(self, step: int) -> Path:
        return self.grids_dir / f"grid_{step:04d}.npy"

    def _border_filename(self) -> Path:
        return self.out_dir / "area_borders.npy"

    def _agents_per_cell_filename(self) -> Path:
        return self.out_dir / "agents_per_cell.npy"

    def write_static(self, model: Any) -> None:
        """Write static information about the model to disk.

        We intentionally write a flattened/serializable summary rather than
        pickling the whole model.
        """
        static = {
            "format_version": 1,
            "height": int(getattr(model, "height", None)),
            "width": int(getattr(model, "width", None)),
            "total_voters": int(len(getattr(model, "voting_agents", []) or [])),
            "num_agents": int(getattr(model, "num_agents", None)),
            "num_colors": int(getattr(model, "num_colors", None)),
            "num_areas": int(getattr(model, "num_areas", None)),
            "voters_per_area": {},
            "step_indexing": {
                "meaning": "post_step",
                "first_recorded_step": 0,
                "step_file": "step_%04d.json",
                "grid_file": "grid_%04d.npy",
            },
            "artifacts": {
                "area_borders": "area_borders.npy",
                "agents_per_cell": "agents_per_cell.npy",
            },
        }

        # Compute voters_per_area from model areas (fast; no datacollector)
        try:
            areas = getattr(model, "areas", None) or []
            for area in areas:
                aid = getattr(area, "unique_id", None)
                if aid is None:
                    continue
                static["voters_per_area"][str(int(aid))] = int(getattr(area, "num_agents", 0) or 0)
        except Exception:
            pass

        # Optional global area id -1
        try:
            ga = getattr(model, "global_area", None)
            if ga is not None:
                static["voters_per_area"][str(int(getattr(ga, "unique_id", -1)))] = int(getattr(ga, "num_agents", 0) or 0)
        except Exception:
            pass

        with open(self.out_dir / "static.json", "w") as f:
            json.dump(static, f, indent=2)

        # Store area border grid (HxW bool) as .npy for fast load
        try:
            borders = get_area_border_grid(model)
            np.save(str(self._border_filename()), np.asarray(borders, dtype=bool))
        except Exception:
            pass

        # Store static agents-per-cell counts (HxW int)
        try:
            apc = get_agents_per_cell_grid(model)
            np.save(str(self._agents_per_cell_filename()), np.asarray(apc, dtype=np.int32))
        except Exception:
            pass

        # Also store static personality information once (used by UI elements)
        self.write_personalities(model)

    def write_personalities(self, model: Any) -> None:
        """Write global + per-area personality distributions (static, step 0).

        This is observables-first: replay does not reconstruct agents; it just
        provides the same attributes the visualization expects.

        File: personalities.json (Schema v1)
        """
        payload: dict[str, Any] = {
            "format_version": 1,
            "personalities": None,
            "global_distribution": None,
            "areas": {},
        }

        try:
            pers = getattr(model, "personalities", None)
            if pers is not None:
                payload["personalities"] = _to_python(pers)
        except Exception:
            payload["personalities"] = None

        try:
            payload["global_distribution"] = _to_python(getattr(model, "personality_distribution", None))
        except Exception:
            payload["global_distribution"] = None

        try:
            areas = getattr(model, "areas", None) or []
            for area in areas:
                aid = getattr(area, "unique_id", getattr(area, "id", None))
                if aid is None:
                    continue
                payload["areas"][str(aid)] = {
                    "num_agents": _to_python(getattr(area, "num_agents", None)),
                    "personality_distribution": _to_python(getattr(area, "personality_distribution", None)),
                }
        except Exception:
            # optional
            pass

        with open(self.out_dir / "personalities.json", "w") as f:
            json.dump(payload, f, indent=2)

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
        try:
            if hasattr(model, "datacollector") and model.datacollector is not None:
                mrep = model.datacollector.get_model_vars_dataframe()
                if len(mrep):
                    last = mrep.iloc[-1].to_dict()
                    # Drop heavy / redundant fields
                    for heavy_key in ("GridColors",):
                        last.pop(heavy_key, None)
                    step_data["model"].update({k: _to_python(v) for k, v in last.items()})
        except Exception:
            step_data["model"].setdefault("note", "datacollector extract failed")

        try:
            if model.areas is not None:
                areas = model.areas
                for area in areas:
                    area_id = int(getattr(area, "unique_id"))
                    step_data["areas"][str(area_id)] = {
                        "VoterTurnout": _to_python(
                            getattr(area, "voter_turnout", None)),
                        "DistToReality": _to_python(
                            getattr(area, "dist_to_reality", None)),
                        "ColorDistribution": _to_python(
                            getattr(area, "color_distribution", None)),
                        "ElectionResults": _to_python(
                            getattr(area, "voted_ordering", None)),
                    }
        except Exception:
            # Fallback to Datacollector (slower):
            # --- Per-area metrics ---
            valid_area_ids = [area.unique_id for area in model.areas]
            # allow optional global area id
            valid_area_ids.append(-1)
            try:
                if hasattr(model,
                           "datacollector") and model.datacollector is not None:
                    adf = model.datacollector.get_agent_vars_dataframe()
                    if adf is not None and len(adf) > 0:
                        # Expected MultiIndex: (Step, AgentID) where AgentID is area.unique_id in our model
                        # We only need the last step's area rows.
                        try:
                            last_step = adf.index.get_level_values(0).max()
                            adf_step = adf.xs(last_step, level=0)
                        except Exception:
                            adf_step = adf

                        for aid, row in adf_step.iterrows():
                            if aid not in valid_area_ids:
                                continue
                            # row is a Series with keys: VoterTurnout, DistToReality, ColorDistribution, ElectionResults
                            step_data["areas"][str(aid)] = {
                                "VoterTurnout": _to_python(
                                    row.get("VoterTurnout")),
                                "DistToReality": _to_python(
                                    row.get("DistToReality")),
                                "ColorDistribution": _to_python(
                                    row.get("ColorDistribution")),
                                "ElectionResults": _to_python(
                                    row.get("ElectionResults")),
                            }
            except Exception:
                pass

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

    def write_meta(self, config: dict, seed: Optional[int] = None) -> None:
        meta = {
            "format_version": 1,
            "schema": {
                "name": "replay_schema_v1",
                "step_indexing": "post_step",
            },
            "config": _to_serializable(config),
            "seed": int(seed) if seed is not None else None,
        }
        with open(self.out_dir / "meta.yaml", "w") as f:
            yaml.safe_dump(meta, f)


def _to_python(obj: Any) -> Any:
    """Convert numpy/scalar types to plain python types for json serialisation."""
    if obj is None:
        return None
    if isinstance(obj, (int, float, str, bool)):
        return obj
    # numpy scalars
    try:
        import numpy as _np
        if isinstance(obj, (_np.generic, _np.ndarray)):
            if getattr(obj, "ndim", 0) == 0:
                return obj.item()
            return obj.tolist()
    except Exception:
        pass
    # pandas types (Series, Timestamp etc.)
    try:
        import pandas as _pd
        if isinstance(obj, (_pd.Series, _pd.Timestamp)):
            return obj.to_json()
    except Exception:
        pass
    # fallback
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)


def _to_serializable(obj: Any) -> Any:
    """Convert objects (including pydantic models) into serializable primitives."""
    # Prefer Pydantic v2 API first to avoid deprecation warnings
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return _to_serializable(obj.model_dump())
        except Exception:
            pass
    # Fallback to Pydantic v1 API if present
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return _to_serializable(obj.dict())
        except Exception:
            pass
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
