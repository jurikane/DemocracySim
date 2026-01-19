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
    """A minimal adapter that mimics mesa.DataCollector for charts + overlays.

    Provides:
    - get_model_vars_dataframe(): DataFrame for ChartModule
    - get_agent_vars_dataframe(): MultiIndex DataFrame for AreaStats/VoterTurnoutElement

    We don't reconstruct simulation; we just replay logged observables.
    """

    def __init__(self):
        self.model_vars: Dict[str, List[Any]] = {}
        self._model_step_count: int = 0
        self._agent_rows: list[dict[str, Any]] = []

    def add_model(self, row: Dict[str, Any]) -> None:
        # Ensure all existing keys receive a value for this step
        for key in list(self.model_vars.keys()):
            if key not in row:
                self.model_vars[key].append(None)
        # Add new keys found in this row; backfill with None for previous steps
        for key, value in row.items():
            if key not in self.model_vars:
                self.model_vars[key] = [None] * self._model_step_count
            self.model_vars[key].append(value)
        self._model_step_count += 1

    def add_area_rows(self, step: int, areas: Dict[str, Any]) -> None:
        # areas: {"<area_id>": {"VoterTurnout":..., ...}}
        for aid, rec in (areas or {}).items():
            area_id = int(aid)
            self._agent_rows.append({
                "Step": int(step),
                "AgentID": int(area_id),
                "VoterTurnout": rec.get("VoterTurnout"),
                "DistToReality": rec.get("DistToReality"),
                "ColorDistribution": rec.get("ColorDistribution"),
                "ElectionResults": rec.get("ElectionResults"),
                "GiniIndex": rec.get("GiniIndex"),
            })

    def get_model_vars_dataframe(self) -> pd.DataFrame:
        # Build DataFrame from model_vars
        return pd.DataFrame(self.model_vars)

    def get_agent_vars_dataframe(self) -> pd.DataFrame:
        if not self._agent_rows:
            return pd.DataFrame()
        df = pd.DataFrame(self._agent_rows)
        # match mesa.DataCollector agent vars format: MultiIndex (Step, AgentID)
        df = df.set_index(["Step", "AgentID"]).sort_index()
        return df


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

    def load_area_borders(self) -> Optional[np.ndarray]:
        static = self.load_static() or {}
        artifacts = static.get("artifacts") if isinstance(static.get("artifacts"), dict) else {}
        f_name = artifacts.get("area_borders", "area_borders.npy")
        p = self.run_dir / f_name
        if p.exists():
            return np.load(str(p))
        return None

    def load_agents_per_cell(self) -> Optional[np.ndarray]:
        static = self.load_static() or {}
        artifacts = static.get("artifacts")
        f_name = artifacts.get("agents_per_cell", "agents_per_cell.npy")
        p = self.run_dir / f_name
        if p.exists():
            return np.load(str(p))
        return None

    def load_agent_strings_per_cell(self) -> Optional[np.ndarray]:
        static = self.load_static() or {}
        artifacts = static.get("artifacts")
        f_name = artifacts.get("agent_strings_per_cell", "agent_strings_per_cell.npy")
        p = self.run_dir / f_name
        if p.exists():
            return np.load(str(p))
        return None


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

        # Expose static voter counts for analysis/UI use
        self.total_voters = int(static.get("total_voters", 0) or 0)
        self.num_voters_per_area = static.get("num_voters_per_area", {}) \
            if isinstance(static.get("voters_per_area"), dict) else {}

        self.grid = mesa.space.SingleGrid(height=self._height, width=self._width, torus=True)
        self.color_cells: list[ColorCell] = []
        borders_arr = self.data.load_area_borders()
        set_borders = False
        if borders_arr is not None and borders_arr.shape == (self._height, self._width):
            set_borders = True
        apc_arr = self.data.load_agents_per_cell()
        apc_str_arr = self.data.load_agent_strings_per_cell()
        apply_apc = False
        if apc_arr is not None and apc_arr.shape == (self._height, self._width):
            apply_apc = True
        for idx, (_, (col, row)) in enumerate(self.grid.coord_iter()):  # In Mesa, coord_iter() yields (contents, (x, y)), i.e. x is column and y is row
            # Create ColorCell with placeholder color 0; will be overridden by snapshots
            cell = ColorCell(unique_id=idx, model=self, pos=(col, row), initial_color=0)
            # Apply static area borders (if present)
            if set_borders:
                cell.is_border_cell = borders_arr[row, col]
            # Apply static agents-per-cell counts (if present)
            if apply_apc:
                n_agents = int(apc_arr[row, col])
                if n_agents > 0:
                    # Create the agents from str info as placeholders
                    agents_str = apc_str_arr[row, col]
                    vote_agents = self._build_agent_stubs(agents_str)
                    cell.agents = vote_agents
            self.color_cells.append(cell)

        # Populate static personality info expected by visualization elements
        self._load_static_personality_info()

        # Areas are not simulated in replay, but AreaPersonalityDists expects area objects.
        self.areas = self._build_area_stubs_from_personalities()
        self.voting_agents = []

        # Apply first snapshot if available
        if len(self.data) > 0:
            self._advance()

    # def _apply_static_borders(self) -> None:
    #     arr = self.data.load_area_borders()
    #     if arr is None:
    #         return
    #     try:
    #         h, w = int(arr.shape[0]), int(arr.shape[1])
    #     except Exception:
    #         return
    #     if h != self._height or w != self._width:
    #         return
    #
    #     # Same coord mapping as colors: arr[y, x]
    #     flat = np.asarray(arr, dtype=bool).T.ravel()
    #     for i, (cell, _pos) in enumerate(self.grid.coord_iter()):
    #         if cell is None:
    #             continue
    #         try:
    #             cell.is_border_cell = bool(flat[i])
    #         except Exception:
    #             pass

    # def _apply_static_agents_per_cell(self) -> None:
    #     """Populate ColorCell.agents with placeholders so the UI can show per-cell counts.
    #
    #     The visualization uses `len(cell.agents)` (via num_agents_in_cell).
    #     We don't reconstruct real VoteAgents; placeholders are fine.
    #     """
    #     arr = self.data.load_agents_per_cell()
    #     if arr is None:
    #         return
    #     try:
    #         h, w = int(arr.shape[0]), int(arr.shape[1])
    #     except Exception:
    #         return
    #     if h != self._height or w != self._width:
    #         return
    #
    #     # Artifacts use arr[y, x]. Mesa stores pos as (x, y).
    #     flat = np.asarray(arr, dtype=np.int32).T.ravel()
    #
    #     # Fill each cell.agents with placeholders
    #     for i, (cell, _pos) in enumerate(self.grid.coord_iter()):
    #         if cell is None:
    #             continue
    #         try:
    #             n = int(flat[i])
    #         except Exception:
    #             n = 0
    #         cell.agents = [None] * n if n > 0 else []

    def _load_static_personality_info(self) -> None:
        payload = self.data.load_static().get("personality_info") or {}
        self.personalities = np.array(payload.get("personalities") or [])
        self.personality_distribution = payload.get("global_distribution") or []
        self._areas_personality_payload = payload.get("areas") or {}

    def _build_area_stubs_from_personalities(self):
        class _AreaStub:
            def __init__(self, unique_id: int, num_agents: int | None, personality_distribution):
                self.unique_id = unique_id
                self.num_agents = int(num_agents) if num_agents is not None else 0
                self.personality_distribution = personality_distribution or []

        stubs = []
        for aid, rec in (self._areas_personality_payload or {}).items():
            aid = int(aid)
            stubs.append(_AreaStub(aid, rec.get("num_agents"),
                                   rec.get("personality_distribution")))
        stubs.sort(key=lambda a: a.unique_id)
        return stubs

    def _build_agent_stubs(self, agents_str):
        class _VoterStub:
            def __init__(self, vote_agent_str: str):
                aid, personality = vote_agent_str.split(": ")
                self.unique_id = int(aid)
                self.personality = personality
                self.assets = "-"

        return [_VoterStub(a_str) for a_str in agents_str.split(", ")]


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
        # Schema v1 supports either old flat shape or new nested shape
        step = int(rec.get("step", idx))

        grid = self.data.load_grid(step)
        if grid is not None:
            self._apply_grid(grid)

        # Model vars for charts; ensure ChartModule labels exist
        model_block = rec.get("model") if isinstance(rec.get("model"), dict) else None
        if model_block is None:
            # legacy: everything except step
            model_block = {k: v for k, v in rec.items() if k != "step"}

        self.datacollector.add_model(model_block)

        # Area rows for overlays
        areas_block = rec.get("areas") if isinstance(rec.get("areas"), dict) else {}
        self.datacollector.add_area_rows(step=step, areas=areas_block)

        self.scheduler.steps = step

    def _apply_grid(self, arr: np.ndarray) -> None:
        """Apply a recorded grid snapshot.

        Snapshot contract:
          - arr has shape (height, width)
          - arr[y, x] is the color at (x, y)

        Mesa's `SingleGrid.coord_iter()` iterates x-major (x in 0-w, y in 0-h).
        To update efficiently (and order-stably), we transpose to (w,h) and
        flatten in C-order, matching coord_iter's order.
        """
        grid = getattr(self, "grid", None)
        if grid is None:
            return

        h, w = int(arr.shape[0]), int(arr.shape[1])

        if int(getattr(grid, "width", 0)) != w or int(getattr(grid, "height", 0)) != h:
            return

        flat = arr.T.ravel()  # (h,w) -> (w,h) x-major flatten

        for i, (cell, _pos) in enumerate(grid.coord_iter()):
            if cell is None:
                continue
            cell.color = int(flat[i])

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
    """Create a ModularServer using ReplayModel with the existing visualization."""

    elements = [make_canvas(appcfg), *make_charts(appcfg)]
    title = "Replay: Participation Model"
    params = {"appcfg": appcfg, "run_dir": str(run_dir)}
    return ModularServer(ReplayModel, elements, title, params)
