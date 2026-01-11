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

    def write_static(self, model: Any) -> None:
        """Write static information about the model to disk.

        We intentionally write a flattened/serializable summary rather than
        pickling the whole model.
        """
        static = {
            "height": int(getattr(model, "height", None)),
            "width": int(getattr(model, "width", None)),
            "num_agents": int(getattr(model, "num_agents", None)),
            "num_colors": int(getattr(model, "num_colors", None)),
            "num_areas": int(getattr(model, "num_areas", None)),
        }
        with open(self.out_dir / "static.json", "w") as f:
            json.dump(static, f, indent=2)

    def append_step(self, step: int, model: Any, grid_snapshot: Optional[np.ndarray] = None) -> None:
        """Append per-step data.

        Contract:
        - Always writes a small JSON in steps/step_XXXX.json with scalar reporters.
        - Optionally writes grids/grid_XXXX.npy when store_grid=True and grid_snapshot is provided.

        IMPORTANT: We intentionally *do not* store large grid-like reporters (e.g. "GridColors")
        inside the JSON because it bloats disk usage and is redundant with the .npy snapshot.
        """
        step_data: dict = {"step": int(step)}

        # Extract model scalars via datacollector if present
        try:
            if hasattr(model, "datacollector") and model.datacollector is not None:
                mrep = model.datacollector.get_model_vars_dataframe()
                if len(mrep):
                    last = mrep.iloc[-1].to_dict()
                    # Drop heavy / redundant fields
                    for heavy_key in ("GridColors",):
                        last.pop(heavy_key, None)
                    step_data.update({k: _to_python(v) for k, v in last.items()})
        except Exception:
            step_data.setdefault("note", "datacollector extract failed")

        # Include minimal per-area metrics if the model provides them (optional)
        # This supports thesis outputs without relying on UI-only collectors.
        try:
            if hasattr(model, "areas") and model.areas is not None:
                area_turnout = {}
                area_gini = {}
                for area in model.areas:
                    aid = getattr(area, "unique_id", getattr(area, "id", None))
                    if aid is None:
                        continue
                    if hasattr(area, "voter_turnout"):
                        area_turnout[str(aid)] = _to_python(getattr(area, "voter_turnout"))
                    if hasattr(area, "gini_index"):
                        area_gini[str(aid)] = _to_python(getattr(area, "gini_index"))
                if area_turnout:
                    step_data.setdefault("areas", {})["turnout"] = area_turnout
                if area_gini:
                    step_data.setdefault("areas", {})["gini"] = area_gini
        except Exception:
            # optional
            pass

        step_file = self._step_filename(step)
        with open(step_file, "w") as f:
            json.dump(step_data, f, indent=2)

        if self.store_grid and grid_snapshot is not None:
            grid_file = self._grid_filename(step)
            arr = np.asarray(grid_snapshot)
            # np.save will append .npy if not given; ensure path has that suffix
            np.save(str(grid_file), arr)

    def flush(self) -> None:
        """Currently a no-op because we write per-step files immediately,
        but provided for API compatibility.
        """
        return

    def write_meta(self, config: dict, seed: Optional[int] = None) -> None:
        # Convert pydantic models or other objects to plain dicts
        meta = {
            "format_version": 1,
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
