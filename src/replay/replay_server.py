from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import mesa
from typing import List, Dict, Any, Optional

from src.config.schema import AppConfig
from mesa.visualization.ModularVisualization import ModularServer
from src.viz.factory import make_canvas, make_charts
from src.agents.color_cell import ColorCell


class _DataCollectorAdapter:
    """A minimal adapter that mimics mesa.DataCollector for charts.

    Provides:
    - model_vars: Dict[str, List[Any]] with one list per reporter label
    - get_model_vars_dataframe(): DataFrame built from model_vars
    - get_agent_vars_dataframe(): empty DataFrame (agent-level not recorded)
    """
    def __init__(self):
        self.model_vars: Dict[str, List[Any]] = {}
        self._step_count: int = 0

    def add(self, row: Dict[str, Any]) -> None:
        # Ensure all existing keys receive a value for this step
        for key in list(self.model_vars.keys()):
            if key not in row:
                self.model_vars[key].append(None)
        # Add new keys found in this row; backfill with None for previous steps
        for key, value in row.items():
            if key not in self.model_vars:
                self.model_vars[key] = [None] * self._step_count
            self.model_vars[key].append(value)
        self._step_count += 1

    def get_model_vars_dataframe(self) -> pd.DataFrame:
        # Build DataFrame from model_vars
        return pd.DataFrame(self.model_vars)

    def get_agent_vars_dataframe(self) -> pd.DataFrame:
        # We don't record agent-level results in replay yet
        return pd.DataFrame()


class _SchedulerStub:
    """Minimal scheduler stub exposing .steps so visualization elements work."""
    def __init__(self):
        self.steps = 0


class ReplayData:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.steps_dir = self.run_dir / "steps"
        self.grids_dir = self.run_dir / "grids"
        self.step_files = sorted(self.steps_dir.glob("step_*.json"))

    def __len__(self):
        return len(self.step_files)

    def load_step(self, index: int) -> Dict[str, Any]:
        sf = self.step_files[index]
        return json.loads(sf.read_text())

    def load_grid(self, step: int) -> Optional[np.ndarray]:
        gf = self.grids_dir / f"grid_{step:04d}.npy"
        if gf.exists():
            return np.load(str(gf))
        return None

    def load_static(self) -> Dict[str, Any]:
        static_path = self.run_dir / "static.json"
        if static_path.exists():
            return json.loads(static_path.read_text())
        return {}


class ReplayModel(mesa.Model):
    """A minimal Mesa model that replays recorded steps using ColorCell agents
    on a SingleGrid, driven entirely by recorded files.
    """
    def __init__(self, appcfg: AppConfig, run_dir: str | Path):
        super().__init__()
        self.appcfg = appcfg
        self.run_dir = Path(run_dir)
        self.scheduler = _SchedulerStub()
        self.datacollector = _DataCollectorAdapter()
        self.data = ReplayData(self.run_dir)
        self._idx = -1
        self.finished = False

        # Build grid and color cells from static info
        static = self.data.load_static()
        self._height = int(static.get("height", getattr(appcfg.model, "height", 1)))
        self._width = int(static.get("width", getattr(appcfg.model, "width", 1)))
        self._num_colors = int(static.get("num_colors", getattr(appcfg.model, "num_colors", 2)))

        self.grid = mesa.space.SingleGrid(height=self._height, width=self._width, torus=True)
        self.color_cells: list[ColorCell] = []
        uid_start = 0
        for idx, (_, (row, col)) in enumerate(self.grid.coord_iter()):
            # Create ColorCell with placeholder color 0; will be overridden by snapshots
            cell = ColorCell(unique_id=uid_start + idx, model=self, pos=(row, col), initial_color=0)
            self.color_cells.append(cell)
        # Areas and agents are not replayed; expose empty collections
        self.areas = []
        self.voting_agents = []
        self.personalities = []
        self.personality_distribution = []

        # Apply first snapshot if available
        if len(self.data) > 0:
            self._advance()

    # --- Properties expected by visualization elements ---
    @property
    def num_agents(self) -> int:
        return 0

    @property
    def num_colors(self) -> int:
        return self._num_colors

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    # --- Replay application helpers ---
    def _apply_index(self, idx: int) -> None:
        rec = self.data.load_step(idx)
        step = int(rec.get("step", idx))
        grid = self.data.load_grid(step)
        if grid is not None:
            self._apply_grid(grid)
        # accumulate reporters (append per-step values)
        model_row = {k: v for k, v in rec.items() if k != "step"}
        model_row["Step"] = step
        self.datacollector.add(model_row)
        self.scheduler.steps = step

    def _apply_grid(self, arr: np.ndarray) -> None:
        """Apply a recorded grid snapshot.

        Snapshot contract:
          - arr has shape (height, width)
          - arr[y, x] is the color at (x, y)

        Mesa's `SingleGrid.coord_iter()` iterates x-major (x in [0..w), y in [0..h)).
        To update efficiently (and order-stably), we transpose to (w,h) and
        flatten in C-order, matching coord_iter's order.
        """
        grid = getattr(self, "grid", None)
        if grid is None:
            return

        try:
            h, w = int(arr.shape[0]), int(arr.shape[1])
        except Exception:
            return

        if int(getattr(grid, "width", 0)) != w or int(getattr(grid, "height", 0)) != h:
            return

        flat = arr.T.ravel()  # (h,w) -> (w,h) x-major flatten

        for i, (cell, _pos) in enumerate(grid.coord_iter()):
            if cell is None:
                continue
            try:
                cell.color = int(flat[i])
            except Exception:
                pass

    def _advance(self) -> None:
        if self.finished:
            return
        next_idx = self._idx + 1
        if 0 <= next_idx < len(self.data):
            self._idx = next_idx
            self._apply_index(self._idx)
            if self._idx == len(self.data) - 1:
                self.finished = True
        else:
            self.finished = True

    def step(self) -> None:
        # Advance to next recorded step if available; do nothing when finished
        self._advance()


def make_replay_server(appcfg: AppConfig, run_dir: Path) -> ModularServer:
    """Create a ModularServer using ReplayModel with the existing visualization.
    """
    elements = [make_canvas(appcfg), *make_charts(appcfg)]
    title = "Replay: Participation Model"
    params = {"appcfg": appcfg, "run_dir": str(run_dir)}
    return ModularServer(ReplayModel, elements, title, params)
