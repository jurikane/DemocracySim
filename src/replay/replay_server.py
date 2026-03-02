from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
import pandas as pd
import mesa
from typing import List, Dict, Any, Optional

from src.config.schema import AppConfig
from mesa.visualization.ModularVisualization import ModularServer
from src.viz.factory import make_canvas, make_charts
from src.agents.color_cell import ColorCell


def _resolve_num_voters_per_area(static: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve canonical voter-count mapping from static metadata."""
    canonical = static.get("num_voters_per_area")
    if "voters_per_area" in static:
        raise ValueError(
            "static.json uses removed key 'voters_per_area'; "
            "use canonical 'num_voters_per_area'."
        )

    if canonical is not None:
        if not isinstance(canonical, dict):
            raise ValueError("static.json key 'num_voters_per_area' must be a JSON object")
        return canonical

    return {}


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
        grid, _grid_step = self.load_grid_with_source(step)
        return grid

    def load_grid_with_source(self, step: int) -> tuple[np.ndarray, int]:
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
                return np.load(str(gf)), int(t)

        # DOE runs with store_grid=false may only persist step-1/last grids.
        # For replay bootstrap at step 0, fall forward to grid_001 when grid_000 is absent.
        if s == 0:
            gf1 = self.grids_dir / (self._grid_pattern % 1)
            if gf1.exists():
                return np.load(str(gf1)), 1

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

    def _artifact_path(self, key: str) -> Path:
        static = self.load_static() or {}
        artifacts = static.get("artifacts") if isinstance(static.get("artifacts"), dict) else {}
        if key not in artifacts:
            raise ValueError(f"static.json missing required artifacts.{key}; run_dir={self.run_dir}")
        p = self.run_dir / str(artifacts[key])
        if not p.exists():
            raise FileNotFoundError(f"Missing {key} artifact: {p}")
        return p

    def load_cell_areas(self) -> pd.DataFrame:
        p = self._artifact_path("cell_areas")
        df = pd.read_parquet(p)
        need = {"x", "y", "area_id"}
        missing = sorted(need - set(df.columns))
        if missing:
            raise KeyError(f"{p.name} missing required columns: {missing}")
        return df

    def load_cell_agents(self) -> pd.DataFrame:
        p = self._artifact_path("cell_agents")
        df = pd.read_parquet(p)
        need = {"x", "y", "area_id", "agent_id", "personality_group_idx"}
        missing = sorted(need - set(df.columns))
        if missing:
            raise KeyError(f"{p.name} missing required columns: {missing}")
        return df

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


def _derive_borders_by_cell(
    *,
    cell_areas_df: pd.DataFrame,
    width: int,
    height: int,
) -> dict[tuple[int, int], bool]:
    """Derive border flags from area occupancy (no static border artifact file)."""
    area_cells: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for r in cell_areas_df.itertuples(index=False):
        x = int(r.x)
        y = int(r.y)
        if x < 0 or x >= int(width) or y < 0 or y >= int(height):
            continue
        area_cells[int(r.area_id)].add((x, y))

    borders: dict[tuple[int, int], bool] = {}
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for cells in area_cells.values():
        if not cells:
            continue
        for x, y in cells:
            for dx, dy in neighbors:
                if (x + dx, y + dy) not in cells:
                    borders[(x, y)] = True
                    break
    return borders


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
        self.num_voters_per_area = _resolve_num_voters_per_area(static)

        self.grid = mesa.space.SingleGrid(height=self._height, width=self._width, torus=True)
        self.color_cells: list[ColorCell] = []

        # Areas are not simulated in replay, but AreaPersonalityGroupDists expects area objects.
        self.areas = self._build_area_stubs_from_personality_groups()
        self.voting_agents = []

        cell_areas_df = self.data.load_cell_areas()
        cell_agents_df = self.data.load_cell_agents()

        borders_by_cell = _derive_borders_by_cell(
            cell_areas_df=cell_areas_df,
            width=self._width,
            height=self._height,
        )

        areas_by_cell: dict[tuple[int, int], list[int]] = defaultdict(list)
        for r in cell_areas_df.itertuples(index=False):
            x = int(r.x)
            y = int(r.y)
            if x < 0 or x >= self._width or y < 0 or y >= self._height:
                continue
            area_id = int(r.area_id)
            if area_id not in areas_by_cell[(x, y)]:
                areas_by_cell[(x, y)].append(area_id)

        agent_ids_by_cell: dict[tuple[int, int], list[int]] = defaultdict(list)
        pg_idx_by_agent_id: dict[int, int] = {}
        for r in cell_agents_df.itertuples(index=False):
            x = int(r.x)
            y = int(r.y)
            if x < 0 or x >= self._width or y < 0 or y >= self._height:
                continue
            agent_id = int(r.agent_id)
            if agent_id not in agent_ids_by_cell[(x, y)]:
                agent_ids_by_cell[(x, y)].append(agent_id)
            if agent_id not in pg_idx_by_agent_id:
                pg_idx_by_agent_id[agent_id] = int(r.personality_group_idx)

        agent_stubs_by_id = self._build_agent_stubs(pg_idx_by_agent_id=pg_idx_by_agent_id)
        self.voting_agents = [agent_stubs_by_id[k] for k in sorted(agent_stubs_by_id)]
        area_by_id = {int(a.unique_id): a for a in self.areas}

        for idx, (_, (col, row)) in enumerate(self.grid.coord_iter()):  # In Mesa, coord_iter() yields (contents, (x, y)), i.e. x is column and y is row
            # Create ColorCell with placeholder color 0; will be overridden by snapshots
            cell = ColorCell(unique_id=idx, model=self, pos=(col, row), initial_color=0)

            cell.is_border_cell = bool(borders_by_cell.get((int(col), int(row)), False))
            area_ids = areas_by_cell.get((int(col), int(row)), [])
            cell.areas = [area_by_id[aid] for aid in area_ids if aid in area_by_id]
            voter_ids = agent_ids_by_cell.get((int(col), int(row)), [])
            cell.agents = [agent_stubs_by_id[aid] for aid in voter_ids if aid in agent_stubs_by_id]

            self.color_cells.append(cell)

        # Replay status for UI diagnostics (recorded step vs grid source step).
        self.replay_recorded_step: int = 0
        self.replay_grid_source_step: int = 0

        # Apply initial pre-election grid snapshot (grid_0000.npy) if available,
        # without advancing recorded step series. This keeps scheduler.steps==0 so
        # UI shows 'Current Step: 0' while the grid matches the true initial state.
        g0, g0_src = self.data.load_grid_with_source(0)
        if g0 is not None:
            self._apply_grid(g0)
            self.replay_grid_source_step = int(g0_src)
            self._initialized_with_grid0 = True
        self.replay_recorded_step = 0
        self.scheduler.steps = 0
        self.scheduler.time = 0

        # Do NOT auto-advance recorded steps here. The first call to step() will
        # advance to the first recorded step (step=1). steps/area_steps color
        # distributions are pre-mutation; grid snapshots are pre-mutation.

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

    def _build_agent_stubs(self, *, pg_idx_by_agent_id: dict[int, int]):
        class _VoterStub:
            def __init__(self, *, agent_id: int, personality_group_idx: int, personality_group: list[int]):
                self.unique_id = int(agent_id)
                self.personality_group_idx = int(personality_group_idx)
                self.personality_group = list(personality_group)
                self.personality = list(personality_group)
                self.assets = "-"

        stubs: dict[int, Any] = {}
        n_groups = int(self.personality_groups.shape[0]) if self.personality_groups.ndim == 2 else 0
        for aid in sorted(pg_idx_by_agent_id):
            pg_idx = int(pg_idx_by_agent_id[aid])
            if 0 <= pg_idx < n_groups:
                personality_group = [int(v) for v in self.personality_groups[pg_idx].tolist()]
            else:
                personality_group = []
            stubs[int(aid)] = _VoterStub(
                agent_id=int(aid),
                personality_group_idx=pg_idx,
                personality_group=personality_group,
            )
        return stubs

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

        grid, grid_src_step = self.data.load_grid_with_source(step)
        if grid is not None:
            self._apply_grid(grid)
        self.replay_recorded_step = int(step)
        self.replay_grid_source_step = int(grid_src_step)

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
