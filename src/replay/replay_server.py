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

        # Detect schema via meta.yaml (v2 runs use parquet + static_v2.json)
        self._schema = self._detect_schema()

        # Load static early so we can honor filename patterns
        self._static = self.load_static()
        self._step_pattern = None
        self._grid_pattern = None
        step_indexing = self._static.get("step_indexing") if isinstance(self._static.get("step_indexing"), dict) else {}
        if isinstance(step_indexing.get("step_file"), str):
            self._step_pattern = step_indexing.get("step_file")
        if isinstance(step_indexing.get("grid_file"), str):
            self._grid_pattern = step_indexing.get("grid_file")

        # Legacy schema uses step JSON discovery; schema v2 uses parquet length.
        self.step_files = self._discover_step_files() if self._schema != "v2" else []
        self._steps_df: Optional[pd.DataFrame] = None
        self._area_steps_df: Optional[pd.DataFrame] = None
        if self._schema == "v2":
            self._load_parquet_tables()

    # -----------------
    # Schema detection
    # -----------------
    def _detect_schema(self) -> str:
        """Return 'v2' if meta.yaml indicates output_schema_v2, else 'legacy'."""
        meta_path = self.run_dir / "meta.yaml"
        if not meta_path.exists():
            return "legacy"
        try:
            import yaml
            meta = yaml.safe_load(meta_path.read_text()) or {}
        except (OSError, ValueError, TypeError):
            return "legacy"
        schema = meta.get("schema") if isinstance(meta.get("schema"), dict) else {}
        name = schema.get("name")
        version = schema.get("version")
        if name == "output_schema_v2" and int(version or 0) == 2:
            return "v2"
        return "legacy"

    def _load_parquet_tables(self) -> None:
        steps_path = self.run_dir / "steps.parquet"
        area_steps_path = self.run_dir / "area_steps.parquet"
        if steps_path.exists():
            self._steps_df = pd.read_parquet(steps_path)
        else:
            self._steps_df = pd.DataFrame()
        if area_steps_path.exists():
            self._area_steps_df = pd.read_parquet(area_steps_path)
        else:
            self._area_steps_df = pd.DataFrame()

    # -----------------
    # Legacy JSON support
    # -----------------
    def _discover_step_files(self):
        files = list(self.steps_dir.glob("step_*.json"))

        def _step_idx(p: Path) -> int:
            stem = p.stem
            suffix = stem.rsplit("_", 1)[-1]
            return int(suffix) if suffix.isdigit() else 10**18

        return sorted(files, key=_step_idx)

    def load_grid(self, step: int) -> Optional[np.ndarray]:
        # Use pattern from static.json/static_v2.json.
        if not self._grid_pattern:
            return None
        gf = self.grids_dir / (self._grid_pattern % int(step))
        if gf.exists():
            return np.load(str(gf))
        return None

    def load_step(self, index: int) -> Dict[str, Any]:
        """Load one step record.

        Returns a unified payload with:
        - step (int)
        - model (dict, snake_case)
        - areas (dict[int, dict], snake_case)
        """
        if self._schema == "v2":
            return self._load_step_v2(index)

        # legacy JSON
        sf = self.step_files[index]
        rec = json.loads(sf.read_text())
        # Legacy shape: either nested {'model':..., 'areas':...} or flat.
        step = int(rec.get("step", index))
        model_block = rec.get("model") if isinstance(rec.get("model"), dict) else None
        if model_block is None:
            model_block = {k: v for k, v in rec.items() if k != "step"}
        areas_block = rec.get("areas") if isinstance(rec.get("areas"), dict) else {}

        # Normalize legacy model_block keys to snake_case
        model_block = {
            k.replace("Collective assets", "collective_assets")
             .replace("Voter turnout globally (in percent)", "turnout")
             .replace("Gini Index (0-100)", "gini_index")
             .replace("Color ", "color_"): v
            for k, v in model_block.items()
        }

        # Keep legacy columns as-is; viz will not use them once migrated.
        return {"step": step, "model": model_block, "areas": areas_block}

    def _load_step_v2(self, index: int) -> Dict[str, Any]:
        steps_df = self._steps_df if self._steps_df is not None else pd.DataFrame()
        area_steps_df = self._area_steps_df if self._area_steps_df is not None else pd.DataFrame()
        if steps_df.empty:
            return {"step": int(index), "model": {}, "areas": {}}

        # Treat 'index' as the sequential step order.
        # steps.parquet is keyed by step, but contract test uses contiguous 0..N-1.
        step = int(steps_df.iloc[index]["step"]) if "step" in steps_df.columns else int(index)
        model_row = steps_df.iloc[index].to_dict()
        model_row.pop("run_seed", None)
        model_row.pop("rule_idx", None)

        areas: Dict[int, Dict[str, Any]] = {}
        if not area_steps_df.empty and "step" in area_steps_df.columns:
            sdf = area_steps_df[area_steps_df["step"].astype(int) == step]
            if not sdf.empty:
                for _, r in sdf.iterrows():
                    aid = int(r.get("area_id", -1))
                    # Pack expanded vectors into python lists for viz convenience
                    area_color = [
                        float(r.get(f"area_color_{i}"))
                        for i in _expanded_range(r, prefix="area_color")
                    ]
                    # Normalize elected_color to snake_case for schema v2
                    if self._schema == "v2":
                        elected_color = [
                            int(r.get(f"elected_color_{i}"))
                            for i in _expanded_range(r, prefix="elected_color")
                        ]
                    else:  # Legacy normalization
                        elected_color = [
                            int(r.get(f"Elected Color {i}"))
                            for i in _expanded_range(r, prefix="Elected Color")
                        ]

                    areas[aid] = {
                        "turnout": float(r.get("turnout", 0.0) or 0.0),
                        "dist_to_reality": float(r.get("dist_to_reality", 0.0) or 0.0),
                        "gini_index": int(r.get("gini_index", 0) or 0),
                        "area_color_distribution": area_color,
                        "elected_color": elected_color,
                    }

        return {"step": step, "model": model_row, "areas": areas}

    def load_static(self) -> Dict[str, Any]:
        # schema v2: static_v2.json
        if self._schema == "v2":
            p = self.run_dir / "static_v2.json"
            if p.exists():
                return json.loads(p.read_text())
            return {}

        # legacy
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

    def load_cell_areas(self) -> Optional[np.ndarray]:
        static = self.load_static() or {}
        artifacts = static.get("artifacts")
        f_name = artifacts.get("cell_areas", "cell_areas.npy")
        p = self.run_dir / f_name
        if p.exists():
            return np.load(str(p))
        return None

    def __len__(self) -> int:
        if self._schema == "v2":
            return 0 if self._steps_df is None else int(len(self._steps_df))
        return len(self.step_files)


def _expanded_range(row: pd.Series, prefix: str) -> List[int]:
    """Return contiguous indices i for which prefix_i exists in the row."""
    cols = [c for c in row.index if isinstance(c, str) and c.startswith(prefix + "_")]
    idxs: List[int] = []
    for c in cols:
        suf = c.split("_", 1)[1]
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

        # Build grid and color cells from static info
        static = self.data.load_static()
        self._height = int(static.get("height", getattr(appcfg.model, "height", 1)))
        self._width = int(static.get("width", getattr(appcfg.model, "width", 1)))
        self._num_colors = int(static.get("num_colors", getattr(appcfg.model, "num_colors", 2)))

        # Populate static personality info expected by visualization elements
        # (must run before we build area stubs)
        self._load_static_personality_info()

        # Expose static voter counts for analysis/UI use
        self.total_voters = int(static.get("total_voters", 0) or 0)
        self.num_voters_per_area = static.get("num_voters_per_area", {}) \
            if isinstance(static.get("voters_per_area"), dict) else {}

        self.grid = mesa.space.SingleGrid(height=self._height, width=self._width, torus=True)
        self.color_cells: list[ColorCell] = []

        # Areas are not simulated in replay, but AreaPersonalityDists expects area objects.
        self.areas = self._build_area_stubs_from_personalities()
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

        # Apply first snapshot if available
        if len(self.data) > 0:
            self._advance()

    def _check_npy_arr(self, arr) -> bool:
        if arr is not None and arr.shape == (self._height, self._width):
            return True
        return False

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
                self.color_distribution = []  # For tooltip compatibility

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
            # For legacy runs, areas_block may be keyed by str; adapter supports int keys.
            if areas_block and all(isinstance(k, str) for k in areas_block.keys()):
                try:
                    coerced = {int(k): v for k, v in areas_block.items()}
                except ValueError:
                    coerced = {}
                # Legacy uses different keys; keep as-is.
                self.datacollector.add_area_rows(step=step, areas=coerced)
            else:
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
