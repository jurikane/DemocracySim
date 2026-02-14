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

    Schema rules (Batch 4):
    - Internally snake_case column names.
    - Visualization elements must query snake_case.
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

    def add_area_rows(self, step: int, areas: Dict[int, Dict[str, Any]]) -> None:
        """Append per-area rows for a specific step.

        `areas` is keyed by area_id (int).
        Values are expected to use snake_case keys.
        """
        for area_id, rec in (areas or {}).items():
            self._agent_rows.append(
                {
                    "step": int(step),
                    "agent_id": int(area_id),
                    "turnout": rec.get("turnout"),
                    "dist_to_reality": rec.get("dist_to_reality"),
                    # expanded vector columns are stored separately in parquet, but
                    # the adapter stores pre-packed vectors for viz.
                    "area_color_distribution": rec.get("area_color_distribution"),
                    "elected_color": rec.get("elected_color"),
                    "gini_index": rec.get("gini_index"),
                }
            )

    def get_model_vars_dataframe(self) -> pd.DataFrame:
        # Build DataFrame from model_vars
        return pd.DataFrame(self.model_vars)

    def get_agent_vars_dataframe(self) -> pd.DataFrame:
        if not self._agent_rows:
            return pd.DataFrame()
        df = pd.DataFrame(self._agent_rows)
        # match mesa.DataCollector agent vars format: MultiIndex (Step, AgentID)
        df = df.set_index(["step", "agent_id"]).sort_index()
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

        # v2-only hard cut: require schema v2
        self._schema = self._detect_schema()
        if self._schema != "v2":
            raise ValueError(f"Replay now requires schema v2 runs; run_dir={self.run_dir}")

        # Load static early so we can honor filename patterns
        self._static = self.load_static()
        self._grid_pattern = None
        step_indexing = self._static.get("step_indexing") if isinstance(self._static.get("step_indexing"), dict) else {}
        if isinstance(step_indexing.get("grid_file"), str):
            self._grid_pattern = step_indexing.get("grid_file")
        if not self._grid_pattern:
            raise ValueError(f"static.json missing step_indexing.grid_file; run_dir={self.run_dir}")

        self.step_files = []
        self._steps_df: Optional[pd.DataFrame] = None
        self._area_steps_df: Optional[pd.DataFrame] = None
        self._load_parquet_tables()

    # -----------------
    # Schema detection
    # -----------------
    def _detect_schema(self) -> str:
        """Return 'v2' if meta.yaml indicates output_schema_v2, else raise.
        Replay is v2-only.
        """
        meta_path = self.run_dir / "meta.yaml"
        if not meta_path.exists():
            raise ValueError(f"Missing meta.yaml; replay requires schema v2 run dirs. run_dir={self.run_dir}")
        try:
            import yaml
            meta = yaml.safe_load(meta_path.read_text()) or {}
        except (OSError, ValueError, TypeError) as e:
            raise ValueError(f"Failed to read meta.yaml; replay requires schema v2. run_dir={self.run_dir}") from e

        schema = meta.get("schema") if isinstance(meta.get("schema"), dict) else {}
        name = schema.get("name")
        version = schema.get("version")
        if name == "output_schema_v2" and int(version or 0) == 2:
            return "v2"
        raise ValueError(
            f"meta.yaml does not describe schema v2 (name={name!r}, version={version!r}); run_dir={self.run_dir}"
        )

    def _load_parquet_tables(self) -> None:
        """Load Parquet tables required for replay (v2-only)."""
        steps_path = self.run_dir / "steps.parquet"
        area_steps_path = self.run_dir / "area_steps.parquet"

        if not steps_path.exists():
            raise FileNotFoundError(f"Missing steps.parquet; run_dir={self.run_dir}")
        if not area_steps_path.exists():
            raise FileNotFoundError(f"Missing area_steps.parquet; run_dir={self.run_dir}")
        self._steps_df = pd.read_parquet(steps_path)
        self._area_steps_df = pd.read_parquet(area_steps_path)

    # -----------------
    # Loading
    # -----------------
    def load_step(self, index: int) -> Dict[str, Any]:
        """Load one step record (v2-only)."""
        return self._load_step_v2(index)

    def load_static(self) -> Dict[str, Any]:
        p = self.run_dir / "static.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing static.json; replay requires schema v2 run dirs. run_dir={self.run_dir}")
        data = json.loads(p.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"static.json must be a JSON object. run_dir={self.run_dir}")
        return data

    def load_grid(self, step: int) -> Optional[np.ndarray]:
        # Use pattern from static.json.
        if not self._grid_pattern:
            raise ValueError(f"static.json missing step_indexing.grid_file; run_dir={self.run_dir}")

        s = int(step)
        # Sparse grid logging support:
        # if grid_s is missing (e.g. grid_interval > 1), carry forward the most
        # recent available snapshot <= s.
        for t in range(s, -1, -1):
            gf = self.grids_dir / (self._grid_pattern % t)
            if gf.exists():
                return np.load(str(gf))

        raise FileNotFoundError(
            f"Missing grid snapshot for step {s} and no earlier fallback found in {self.grids_dir}"
        )

    def _load_step_v2(self, index: int) -> Dict[str, Any]:
        steps_df = self._steps_df if self._steps_df is not None else pd.DataFrame()
        area_steps_df = self._area_steps_df if self._area_steps_df is not None else pd.DataFrame()
        if steps_df.empty:
            raise ValueError(f"steps.parquet is empty; run_dir={self.run_dir}")
        if area_steps_df.empty:
            raise ValueError(f"area_steps.parquet is empty; run_dir={self.run_dir}")
        if "step" not in steps_df.columns:
            raise KeyError("steps.parquet missing required column: step")
        if "step" not in area_steps_df.columns:
            raise KeyError("area_steps.parquet missing required column: step")
        if "area_id" not in area_steps_df.columns:
            raise KeyError("area_steps.parquet missing required column: area_id")

        if not any(isinstance(c, str) and c.startswith("area_color_") for c in area_steps_df.columns):
            raise KeyError("area_steps.parquet missing area_color_* columns")
        if not any(isinstance(c, str) and c.startswith("elected_color_") for c in area_steps_df.columns):
            raise KeyError("area_steps.parquet missing elected_color_* columns")

        # 'index' is the sequential position (0...len-1). The recorded 'step' value
        # is taken from parquet (schema v2 is 1-based).
        model_row_series = steps_df.iloc[int(index)]
        step = int(model_row_series["step"])
        model_row = model_row_series.to_dict()
        model_row.pop("run_seed", None)
        model_row.pop("rule_idx", None)

        areas: Dict[int, Dict[str, Any]] = {}
        if not area_steps_df.empty and "step" in area_steps_df.columns:
            sdf = area_steps_df[area_steps_df["step"].astype(int) == step]
            if sdf.empty:
                raise ValueError(f"area_steps.parquet has no rows for step={step}")
            for _, r in sdf.iterrows():
                aid = int(r["area_id"])
                # Pack expanded vectors into python lists for viz convenience
                area_color = [
                    float(r[f"area_color_{i}"])
                    for i in _expanded_range(r, prefix="area_color")
                ]
                elected_color = [
                    int(r[f"elected_color_{i}"])
                    for i in _expanded_range(r, prefix="elected_color")
                ]
                if not area_color:
                    raise KeyError("area_steps.parquet missing area_color_* values")
                if not elected_color:
                    raise KeyError("area_steps.parquet missing elected_color_* values")

                areas[aid] = {
                    "turnout": float(r["turnout"]),
                    "dist_to_reality": float(r["dist_to_reality"]),
                    "gini_index": int(r["gini_index"]),
                    "area_color_distribution": area_color,
                    "elected_color": elected_color,
                }

        return {"step": step, "model": model_row, "areas": areas}

    def load_area_borders(self) -> Optional[np.ndarray]:
        static = self.load_static() or {}
        artifacts = static.get("artifacts") if isinstance(static.get("artifacts"), dict) else {}
        f_name = artifacts.get("area_borders", "area_borders.npy")
        p = self.run_dir / f_name
        if not p.exists():
            raise FileNotFoundError(f"Missing area_borders artifact: {p}")
        return np.load(str(p))

    def load_agents_per_cell(self) -> Optional[np.ndarray]:
        static = self.load_static() or {}
        artifacts = static.get("artifacts")
        f_name = artifacts.get("agents_per_cell", "agents_per_cell.npy")
        p = self.run_dir / f_name
        if not p.exists():
            raise FileNotFoundError(f"Missing agents_per_cell artifact: {p}")
        return np.load(str(p))

    def load_agent_strings_per_cell(self) -> Optional[np.ndarray]:
        static = self.load_static() or {}
        artifacts = static.get("artifacts")
        f_name = artifacts.get("agent_strings_per_cell", "agent_strings_per_cell.npy")
        p = self.run_dir / f_name
        if not p.exists():
            raise FileNotFoundError(f"Missing agent_strings_per_cell artifact: {p}")
        return np.load(str(p))

    def load_cell_areas(self) -> Optional[np.ndarray]:
        static = self.load_static() or {}
        artifacts = static.get("artifacts")
        f_name = artifacts.get("cell_areas", "cell_areas.npy")
        p = self.run_dir / f_name
        if not p.exists():
            raise FileNotFoundError(f"Missing cell_areas artifact: {p}")
        return np.load(str(p))

    def __len__(self) -> int:
        if self._schema == "v2":
            return 0 if self._steps_df is None else int(len(self._steps_df))
        return len(self.step_files)


def _expanded_range(row: pd.Series, prefix: str) -> List[int]:
    """Return contiguous indices i for which prefix_i exists in the row.

    Example: prefix='area_color' matches columns ['area_color_0', 'area_color_1', ...].
    """
    cols = [c for c in row.index if isinstance(c, str) and c.startswith(prefix + "_")]
    idxs: List[int] = []
    for c in cols:
        suf = c.rsplit("_", 1)[-1]
        try:
            idxs.append(int(suf))
        except ValueError:
            continue
    if not idxs:
        return []
    return list(range(0, max(idxs) + 1))


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
        self._initialized_with_grid0: bool = False

        # Build grid and color cells from static info
        static = self.data.load_static()
        self._height = int(static["height"])
        self._width = int(static["width"])
        self._num_colors = int(static["num_colors"])

        # Populate static personality_group info expected by visualization elements
        # (must run before we build area stubs)
        self._load_static_personality_group_info()

        # Expose static voter counts for analysis/UI use
        self.total_voters = int(static.get("total_voters", 0) or 0)
        self.num_voters_per_area = static.get("num_voters_per_area", {}) \
            if isinstance(static.get("voters_per_area"), dict) else {}

        self.grid = mesa.space.SingleGrid(height=self._height, width=self._width, torus=True)
        self.color_cells: list[ColorCell] = []

        # Areas are not simulated in replay, but AreaPersonalityGroupDists expects area objects.
        self.areas = self._build_area_stubs_from_personality_groups()
        self.voting_agents = []

        # Load npy static data if present
        borders_arr = self.data.load_area_borders()
        set_borders = self._check_npy_arr(borders_arr)

        # New static artifact: per-cell area assignments as comma-separated area ids.
        cell_areas_arr = self.data.load_cell_areas()
        set_cell_areas = self._check_npy_arr(cell_areas_arr)

        apc_arr = self.data.load_agents_per_cell()
        apply_apc = self._check_npy_arr(apc_arr)
        apc_str_arr = self.data.load_agent_strings_per_cell()
        set_a_strings = self._check_npy_arr(apc_str_arr)

        for idx, (_, (col, row)) in enumerate(self.grid.coord_iter()):  # In Mesa, coord_iter() yields (contents, (x, y)), i.e. x is column and y is row
            # Create ColorCell with placeholder color 0; will be overridden by snapshots
            cell = ColorCell(unique_id=idx, model=self, pos=(col, row), initial_color=0)

            # Apply static area borders (if present)
            if set_borders:
                cell.is_border_cell = borders_arr[row, col]

            # Apply static agents-per-cell information (if present)
            if apply_apc:
                n_agents = int(apc_arr[row, col])
                if n_agents > 0 and set_a_strings:
                    # Create the agents from str info as placeholders
                    agents_str = apc_str_arr[row, col]
                    vote_agents = self._build_agent_stubs(agents_str)
                    self.voting_agents.extend(vote_agents)
                    cell.agents = vote_agents

            # Apply static area assignments (if present)
            if set_cell_areas:
                area_str = str(cell_areas_arr[row, col] or "").strip()
                try:
                    area_ids = [int(i) for i in area_str.split(", ") if i != ""]
                except ValueError:
                    area_ids = []
                cell.areas = [a for a in self.areas if a.unique_id in set(area_ids)]

            self.color_cells.append(cell)

        # Apply initial pre-election grid snapshot (grid_0000.npy) if available,
        # without advancing recorded step series. This keeps scheduler.steps==0 so
        # UI shows 'Current Step: 0' while the grid matches the true initial state.
        g0 = self.data.load_grid(0)
        if g0 is not None:
            self._apply_grid(g0)
            self._initialized_with_grid0 = True

        # Do NOT auto-advance recorded steps here. The first call to step() will
        # advance to the first recorded step (step=1). steps/area_steps color
        # distributions are pre-mutation; grid snapshots are pre-mutation.

    def _check_npy_arr(self, arr) -> bool:
        if arr is not None and arr.shape == (self._height, self._width):
            return True
        return False

    def _load_static_personality_group_info(self) -> None:
        payload = self.data.load_static().get("personality_group_info") or {}
        self.personality_groups = np.array(payload.get("personality_groups") or [])
        self.personality_group_distribution = payload.get("global_distribution") or []
        self._areas = payload.get("areas") or {}

    def _build_area_stubs_from_personality_groups(self):
        class _AreaStub:
            def __init__(self, unique_id: int, num_agents: int | None, personality_group_distribution):
                self.unique_id = unique_id
                self.num_agents = int(num_agents) if num_agents is not None else 0
                self.personality_group_distribution = personality_group_distribution or []
                self.color_distribution = []  # For tooltip compatibility

        stubs = []
        for aid, rec in (self._areas or {}).items():
            aid = int(aid)
            stubs.append(_AreaStub(aid, rec.get("num_agents"),
                                   rec.get("personality_group_distribution")))
        stubs.sort(key=lambda a: a.unique_id)
        return stubs

    def _build_agent_stubs(self, agents_str):
        class _VoterStub:
            def __init__(self, vote_agent_str: str):
                aid, personality_group = vote_agent_str.split(": ")
                self.unique_id = int(aid)
                self.personality_group_idx = "-" # Placeholder; the idx is not yet available in static info.
                self.personality_group = personality_group
                self.personality = "-"  # Placeholder; the actual personality vector is not yet available in static info.
                self.assets = "-"

        return [_VoterStub(a_str) for a_str in agents_str.split(", ")]

    # --- Properties expected by visualization elements ---
    @property
    def num_agents(self) -> int:
        return len(self.voting_agents)

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

        # Model vars for charts (snake_case)
        model_block = rec.get("model") if isinstance(rec.get("model"), dict) else {}
        self.datacollector.add_model(model_block)

        # Area rows for viz
        areas_block = rec.get("areas")
        if isinstance(areas_block, dict):
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
        grid = self.grid

        h, w = int(arr.shape[0]), int(arr.shape[1])

        if int(grid.width) != w or int(grid.height) != h:
            raise ValueError(f"Grid snapshot shape {(h, w)} does not match grid {(grid.height, grid.width)}.")

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
        # Advance to next recorded step if available; do nothing when finished.
        # The initial state (step 0) is the pre-election grid snapshot only.
        self._advance()


def make_replay_server(appcfg: AppConfig, run_dir: Path) -> ModularServer:
    """Create a ModularServer using ReplayModel with the existing visualization."""

    elements = [make_canvas(appcfg), *make_charts(appcfg)]
    title = "Replay: Participation Model"
    params = {"appcfg": appcfg, "run_dir": str(run_dir)}
    return ModularServer(ReplayModel, elements, title, params)
