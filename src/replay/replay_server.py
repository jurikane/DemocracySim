from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import mesa
import numpy as np
import pandas as pd
import yaml

from src.config.schema import AppConfig
from mesa.visualization.ModularVisualization import ModularServer
from src.viz.factory import make_canvas, make_charts
from src.agents.color_cell import ColorCell
from src.utils.metrics import gini_index_0_100


def _resolve_num_voters_per_area(static: dict[str, object]) -> dict[str, object]:
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
        self.model_vars: dict[str, list[object]] = {}
        self._model_step_count = 0
        self._agent_rows: list[dict[str, object]] = []
        self._model_df: pd.DataFrame | None = None
        self._agent_df: pd.DataFrame | None = None

    def add_model(self, row: dict[str, object]) -> None:
        for key in self.model_vars:
            if key not in row:
                self.model_vars[key].append(None)
        # Add new keys found in this row; backfill with None for previous steps
        for key, value in row.items():
            if key not in self.model_vars:
                self.model_vars[key] = [None] * self._model_step_count
            self.model_vars[key].append(value)
        self._model_step_count += 1
        self._model_df = None

    def add_area_rows(self, step: int, areas: dict[int, dict[str, object]]) -> None:
        """Append per-area rows for a specific step.

        `areas` is keyed by area_id (int).
        Values are expected to use snake_case keys.
        """
        for area_id, rec in areas.items():
            self._agent_rows.append(
                {
                    "step": step,
                    "agent_id": area_id,
                    "turnout": rec["turnout"],
                    "quality_distance": rec["quality_distance"],
                    "dist_to_reality": rec["dist_to_reality"],
                    "puzzle_distance": rec["puzzle_distance"],
                    "area_color_distribution": rec["area_color_distribution"],
                    "puzzle_color_distribution": rec["puzzle_color_distribution"],
                    "elected_color": rec["elected_color"],
                    "gini_index": rec["gini_index"],
                }
            )
        self._agent_df = None

    def get_model_vars_dataframe(self) -> pd.DataFrame:
        if self._model_df is None:
            self._model_df = pd.DataFrame(self.model_vars)
        return self._model_df

    def get_agent_vars_dataframe(self) -> pd.DataFrame:
        if self._agent_df is not None:
            return self._agent_df
        if not self._agent_rows:
            return pd.DataFrame()
        df = pd.DataFrame(self._agent_rows)
        self._agent_df = df.set_index(["step", "agent_id"]).sort_index()
        return self._agent_df


class _SchedulerStub:
    """Minimal scheduler stub exposing .steps so visualization elements work."""

    def __init__(self):
        self.steps = 0


class ReplayData:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.grids_dir = self.run_dir / "grids"
        self._meta = self._load_meta()
        self._detect_schema(self._meta)
        self._static = self._read_static()
        run_cfg = self._meta.get("run") or {}
        if not isinstance(run_cfg, dict):
            raise ValueError(f"meta.yaml has invalid run section. run_dir={self.run_dir}")
        self._quality_target_mode = run_cfg.get("quality_target_mode", "reality").strip().lower()
        self._grid_pattern = self._static["step_indexing"]["grid_file"]
        self._artifacts = self._static["artifacts"]

        self._steps_df: pd.DataFrame
        self._area_steps_df: pd.DataFrame
        self._gini_dissatisfaction_by_step: dict[int, float] = {}
        self._quality_distance_by_step: dict[int, float] = {}
        self._group_metrics_by_step: dict[int, dict[str, list[float]]] = {}
        self._cell_areas_df: pd.DataFrame | None = None
        self._cell_agents_df: pd.DataFrame | None = None
        self._area_color_columns: list[str] = []
        self._elected_color_columns: list[str] = []
        self._puzzle_color_columns: list[str] = []
        self._group_outcome_columns: list[str] = []
        self._load_parquet_tables()

    @property
    def quality_target_mode(self) -> str:
        return self._quality_target_mode

    def _load_meta(self) -> dict[str, object]:
        meta_path = self.run_dir / "meta.yaml"
        if not meta_path.exists():
            raise ValueError(f"Missing meta.yaml; replay requires schema v2/v3 run dirs. run_dir={self.run_dir}")
        try:
            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError, TypeError) as e:
            raise ValueError(f"Failed to read meta.yaml; replay requires schema v2/v3. run_dir={self.run_dir}") from e
        if not isinstance(meta, dict):
            raise ValueError(f"meta.yaml must be a mapping. run_dir={self.run_dir}")
        return meta

    def _read_static(self) -> dict[str, object]:
        path = self.run_dir / "static.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing static.json; replay requires schema v2/v3 run dirs. run_dir={self.run_dir}")
        static = json.loads(path.read_text())
        if not isinstance(static, dict):
            raise ValueError(f"static.json must be a JSON object. run_dir={self.run_dir}")
        step_indexing = static.get("step_indexing")
        if not isinstance(step_indexing, dict) or not isinstance(step_indexing.get("grid_file"), str):
            raise ValueError(f"static.json missing step_indexing.grid_file; run_dir={self.run_dir}")
        artifacts = static.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError(f"static.json missing artifacts map; run_dir={self.run_dir}")
        return static

    # -----------------
    # Schema detection
    # -----------------
    def _detect_schema(self, meta: dict[str, object]) -> None:
        schema = meta.get("schema", {})
        if not isinstance(schema, dict):
            raise ValueError(f"meta.yaml has invalid schema section. run_dir={self.run_dir}")
        name = schema.get("name")
        version = schema.get("version")
        if (name, version) in {("output_schema_v2", 2), ("output_schema_v3", 3)}:
            return
        raise ValueError(
            f"meta.yaml does not describe supported schema v2/v3 (name={name!r}, version={version!r}); run_dir={self.run_dir}"
        )

    def _load_parquet_tables(self) -> None:
        """Load Parquet tables required for replay."""
        steps_path = self.run_dir / "steps.parquet"
        area_steps_path = self.run_dir / "area_steps.parquet"
        agents_path = self.run_dir / "agents.parquet"

        if not steps_path.exists():
            raise FileNotFoundError(f"Missing steps.parquet; run_dir={self.run_dir}")
        if not area_steps_path.exists():
            raise FileNotFoundError(f"Missing area_steps.parquet; run_dir={self.run_dir}")
        self._steps_df = pd.read_parquet(steps_path)
        self._area_steps_df = pd.read_parquet(area_steps_path)
        self._area_color_columns = _expanded_columns(self._area_steps_df.columns, prefix="area_color")
        self._elected_color_columns = _expanded_columns(self._area_steps_df.columns, prefix="elected_color")
        self._puzzle_color_columns = _expanded_columns(self._area_steps_df.columns, prefix="puzzle_color")
        self._group_outcome_columns = _expanded_columns(self._area_steps_df.columns, prefix="group_outcome_distance")

        if "step" not in self._steps_df.columns:
            raise KeyError("steps.parquet missing required column: step")
        if "step" not in self._area_steps_df.columns:
            raise KeyError("area_steps.parquet missing required column: step")
        if "area_id" not in self._area_steps_df.columns:
            raise KeyError("area_steps.parquet missing required column: area_id")
        if not self._area_color_columns:
            raise KeyError("area_steps.parquet missing area_color_* columns")
        if not self._elected_color_columns:
            raise KeyError("area_steps.parquet missing elected_color_* columns")

        self._build_derived_model_metric_overrides(agents_path=agents_path)

    def _build_derived_model_metric_overrides(self, *, agents_path: Path) -> None:
        area_steps_df = self._area_steps_df
        if not area_steps_df.empty:
            value_col = "puzzle_distance" if self._quality_target_mode == "puzzle" else "dist_to_reality"
            if value_col in area_steps_df.columns and "eligible_voters" in area_steps_df.columns:
                for step, block in area_steps_df.groupby("step", sort=True):
                    weights = block["eligible_voters"].to_numpy(dtype=float)
                    values = block[value_col].to_numpy(dtype=float)
                    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
                    denom = np.sum(weights[finite])
                    if denom > 0.0:
                        self._quality_distance_by_step[step] = np.sum(values[finite] * weights[finite]) / denom

        if not agents_path.exists():
            return
        agents_df = pd.read_parquet(
            agents_path,
            columns=[
                "step",
                "agent_id",
                "assets",
                "dissatisfaction_value",
                "personality_group_idx",
                "eligible_for_election",
                "participating",
            ],
        )
        if agents_df.empty or "step" not in agents_df.columns or "dissatisfaction_value" not in agents_df.columns:
            return
        grouped = agents_df.groupby("step", sort=True)["dissatisfaction_value"]
        self._gini_dissatisfaction_by_step = {
            step: gini_index_0_100(vals.to_numpy(dtype=float))
            for step, vals in grouped
        }
        self._build_group_metric_vectors(agents_df=agents_df, area_steps_df=area_steps_df)

    def _build_group_metric_vectors(self, *, agents_df: pd.DataFrame, area_steps_df: pd.DataFrame) -> None:
        if agents_df.empty:
            return

        personality_info = self._static.get("personality_group_info") or {}
        if not isinstance(personality_info, dict):
            personality_info = {}
        n_groups = len(personality_info.get("personality_groups") or [])
        if n_groups <= 0 and "personality_group_idx" in agents_df.columns:
            n_groups = agents_df["personality_group_idx"].max() + 1
        if n_groups <= 0:
            return

        agent_area_df = self.load_cell_agents()[["agent_id", "area_id"]].drop_duplicates(subset=["agent_id"], keep="first")
        agents_df = agents_df.merge(agent_area_df, on="agent_id", how="left")

        static_area_group_counts: dict[int, list[float]] = {}
        for aid, rec in (personality_info.get("areas") or {}).items():
            area_id = int(aid)
            num_agents = rec.get("num_agents", 0)
            distribution = rec.get("personality_group_distribution") or []
            static_area_group_counts[area_id] = [num_agents * value for value in distribution]

        step_area_rows = {
            step: list(block[["area_id", *self._group_outcome_columns]].itertuples(index=False, name=None))
            for step, block in area_steps_df.groupby("step", sort=True)
        }

        for step, block in agents_df.groupby("step", sort=True):
            metrics = {
                "group_turnout": [float("nan")] * n_groups,
                "group_mean_assets_share": [float("nan")] * n_groups,
                "group_mean_dissatisfaction": [float("nan")] * n_groups,
                "group_outcome_distance": [float("nan")] * n_groups,
            }
            group_blocks = {group_idx: group_block for group_idx, group_block in block.groupby("personality_group_idx", sort=False)}
            group_mean_assets = [float("nan")] * n_groups
            step_rows = step_area_rows.get(step)
            for g in range(n_groups):
                g_block = group_blocks.get(g)
                if g_block is None:
                    continue
                elig = g_block[g_block["eligible_for_election"]]
                if not elig.empty:
                    metrics["group_turnout"][g] = 100.0 * elig["participating"].sum() / len(elig)
                group_mean_assets[g] = g_block["assets"].mean()
                metrics["group_mean_dissatisfaction"][g] = g_block["dissatisfaction_value"].mean()
                if step_rows is not None and self._group_outcome_columns:
                    weighted_sum = 0.0
                    total_weight = 0.0
                    for row in step_rows:
                        area_id = row[0]
                        area_counts = static_area_group_counts.get(area_id, [])
                        if len(area_counts) <= g:
                            continue
                        weight = area_counts[g]
                        if weight <= 0.0:
                            continue
                        value = row[g + 1]
                        if not np.isfinite(value):
                            continue
                        weighted_sum += value * weight
                        total_weight += weight
                    if total_weight > 0.0:
                        metrics["group_outcome_distance"][g] = weighted_sum / total_weight
            total_group_mean_assets = np.nansum(group_mean_assets)
            if total_group_mean_assets > 0.0:
                metrics["group_mean_assets_share"] = [
                    value / total_group_mean_assets if np.isfinite(value) else float("nan")
                    for value in group_mean_assets
                ]
            self._group_metrics_by_step[step] = metrics

    # -----------------
    # Loading
    # -----------------
    def load_step(self, index: int) -> dict[str, object]:
        """Load one recorded step from parquet-backed replay data."""
        return self._load_step_record(index)

    def load_static(self) -> dict[str, object]:
        return self._static

    def load_grid(self, step: int) -> np.ndarray:
        grid, _grid_step = self.load_grid_with_source(step)
        return grid

    def load_grid_with_source(self, step: int) -> tuple[np.ndarray, int]:
        # Use pattern from static.json.
        if not self._grid_pattern:
            raise ValueError(f"static.json missing step_indexing.grid_file; run_dir={self.run_dir}")

        s = step
        # Sparse grid logging support:
        # if grid_s is missing (e.g. grid_interval > 1), carry forward the most
        # recent available snapshot <= s.
        for t in range(s, -1, -1):
            gf = self.grids_dir / (self._grid_pattern % t)
            if gf.exists():
                return np.load(gf), t

        # DOE runs with store_grid=false may only persist step-1/last grids.
        # For replay bootstrap at step 0, fall forward to grid_001 when grid_000 is absent.
        if s == 0:
            gf1 = self.grids_dir / (self._grid_pattern % 1)
            if gf1.exists():
                return np.load(gf1), 1

        raise FileNotFoundError(
            f"Missing grid snapshot for step {s} and no earlier fallback found in {self.grids_dir}"
        )

    def _load_step_record(self, index: int) -> dict[str, object]:
        if self._steps_df.empty:
            raise ValueError(f"steps.parquet is empty; run_dir={self.run_dir}")
        if self._area_steps_df.empty:
            raise ValueError(f"area_steps.parquet is empty; run_dir={self.run_dir}")

        step = self._steps_df["step"].iat[index]
        model_row = {column: self._steps_df[column].iat[index] for column in self._steps_df.columns}
        model_row.pop("run_seed", None)
        model_row.pop("rule_idx", None)
        model_row.setdefault("gini_dissatisfaction", self._gini_dissatisfaction_by_step.get(step, float("nan")))
        model_row.setdefault("quality_distance", self._quality_distance_by_step.get(step, float("nan")))
        for key, values in self._group_metrics_by_step.get(step, {}).items():
            model_row.setdefault(key, values)

        sdf = self._area_steps_df[self._area_steps_df["step"] == step]
        if sdf.empty:
            raise ValueError(f"area_steps.parquet has no rows for step={step}")
        areas: dict[int, dict[str, object]] = {}
        area_colors = sdf[self._area_color_columns].to_numpy(dtype=float).tolist()
        elected_colors = sdf[self._elected_color_columns].to_numpy(dtype=int).tolist()
        puzzle_colors = sdf[self._puzzle_color_columns].to_numpy(dtype=float).tolist() if self._puzzle_color_columns else []
        for idx, row in enumerate(sdf.itertuples(index=False)):
            if self._quality_target_mode == "puzzle" and not np.isfinite(row.puzzle_distance):
                raise ValueError("Non-finite puzzle_distance in puzzle-mode replay step rows")

            areas[row.area_id] = {
                "turnout": row.turnout,
                "quality_distance": row.puzzle_distance if self._quality_target_mode == "puzzle" else row.dist_to_reality,
                "dist_to_reality": row.dist_to_reality,
                "puzzle_distance": row.puzzle_distance,
                "gini_index": row.gini_index,
                "area_color_distribution": area_colors[idx],
                "puzzle_color_distribution": puzzle_colors[idx] if puzzle_colors else [],
                "elected_color": elected_colors[idx],
            }

        return {"step": step, "model": model_row, "areas": areas}

    def _artifact_path(self, key: str) -> Path:
        try:
            rel_path = self._artifacts[key]
        except KeyError as e:
            raise ValueError(f"static.json missing required artifacts.{key}; run_dir={self.run_dir}") from e
        path = self.run_dir / rel_path
        if not path.exists():
            raise FileNotFoundError(f"Missing {key} artifact: {path}")
        return path

    def load_cell_areas(self) -> pd.DataFrame:
        if self._cell_areas_df is not None:
            return self._cell_areas_df

        df = pd.read_parquet(self._artifact_path("cell_areas"))
        required = {"x", "y", "area_id"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise KeyError(f"static_cell_areas.parquet missing required columns: {missing}")
        self._cell_areas_df = df
        return df

    def load_cell_agents(self) -> pd.DataFrame:
        if self._cell_agents_df is not None:
            return self._cell_agents_df

        df = pd.read_parquet(self._artifact_path("cell_agents"))
        required = {"x", "y", "area_id", "agent_id", "personality_group_idx"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise KeyError(f"static_cell_agents.parquet missing required columns: {missing}")
        self._cell_agents_df = df
        return df

    def __len__(self) -> int:
        return len(self._steps_df)


def _expanded_columns(columns, *, prefix: str) -> list[str]:
    names = [name for name in columns if name.startswith(prefix + "_")]
    return sorted(names, key=lambda name: int(name.rsplit("_", 1)[-1]))


def _derive_borders_by_cell(
    *,
    cell_areas_df: pd.DataFrame,
    width: int,
    height: int,
) -> dict[tuple[int, int], bool]:
    """Derive border flags from area occupancy (no static border artifact file)."""
    area_cells: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for row in cell_areas_df.itertuples(index=False):
        x = row.x
        y = row.y
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        area_cells[row.area_id].add((x, y))

    borders: dict[tuple[int, int], bool] = {}
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for cells in area_cells.values():
        for x, y in cells:
            if any((x + dx, y + dy) not in cells for dx, dy in neighbors):
                borders[(x, y)] = True
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
        self.quality_target_mode = self.data.quality_target_mode
        self._idx = -1
        self.finished = False

        # Build grid and color cells from static info
        static = self.data.load_static()
        self._height = static["height"]
        self._width = static["width"]
        self._num_colors = static["num_colors"]

        # Populate static personality_group info expected by viz elements
        # (must run before we build area stubs)
        self._load_static_personality_group_info()
        # Expose static voter counts for analysis/UI use
        self.total_voters = static.get("total_voters", 0)
        self.num_voters_per_area = _resolve_num_voters_per_area(static)

        self.grid = mesa.space.SingleGrid(height=self._height,
                                          width=self._width,
                                          torus=True)
        self.color_cells: list[ColorCell] = []
        # Areas aren't simulated, but AreaPersonalityGroupDists expects area objects.
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
            x = r.x
            y = r.y
            if x < 0 or x >= self._width or y < 0 or y >= self._height:
                continue
            area_id = r.area_id
            if area_id not in areas_by_cell[(x, y)]:
                areas_by_cell[(x, y)].append(area_id)

        agent_ids_by_cell: dict[tuple[int, int], list[int]] = defaultdict(list)
        pg_idx_by_agent_id: dict[int, int] = {}
        for r in cell_agents_df.itertuples(index=False):
            x = r.x
            y = r.y
            if x < 0 or x >= self._width or y < 0 or y >= self._height:
                continue
            agent_id = r.agent_id
            if agent_id not in agent_ids_by_cell[(x, y)]:
                agent_ids_by_cell[(x, y)].append(agent_id)
            if agent_id not in pg_idx_by_agent_id:
                pg_idx_by_agent_id[agent_id] = r.personality_group_idx

        agent_stubs_by_id = self._build_agent_stubs(pg_idx_by_agent_id=pg_idx_by_agent_id)
        self.voting_agents = [agent_stubs_by_id[k] for k in sorted(agent_stubs_by_id)]
        area_by_id = {a.unique_id: a for a in self.areas}

        for idx, (_, (col, row)) in enumerate(self.grid.coord_iter()):  # In Mesa, coord_iter() yields (contents, (x, y)), i.e. x is column and y is row
            # Create ColorCell with placeholder color 0; will be overridden by snapshots
            cell = ColorCell(unique_id=idx, model=self, pos=(col, row), initial_color=0)

            cell.is_border_cell = borders_by_cell.get((col, row), False)
            area_ids = areas_by_cell.get((col, row), [])
            cell.areas = [area_by_id[aid] for aid in area_ids if aid in area_by_id]
            voter_ids = agent_ids_by_cell.get((col, row), [])
            cell.agents = [agent_stubs_by_id[aid] for aid in voter_ids if aid in agent_stubs_by_id]

            self.color_cells.append(cell)

        # Replay status for UI diagnostics (recorded step vs grid source step).
        self.replay_recorded_step: int = 0
        self.replay_grid_source_step: int = 0

        # Apply initial pre-election grid snapshot (grid_0000.npy) if available,
        # without advancing recorded step series. This keeps scheduler.steps==0 so
        # UI shows 'Current Step: 0' while the grid matches the true initial state.
        g0, g0_src = self.data.load_grid_with_source(0)
        self._apply_grid(g0)
        self.replay_grid_source_step = g0_src
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
        self._areas = {
            int(area_id): record
            for area_id, record in (payload.get("areas") or {}).items()
        }

    def _build_area_stubs_from_personality_groups(self):
        class _AreaStub:
            def __init__(self, unique_id: int, num_agents: int | None, personality_group_distribution):
                self.unique_id = unique_id
                self.num_agents = num_agents if num_agents is not None else 0
                self.personality_group_distribution = personality_group_distribution or []
                self.color_distribution = []  # For tooltip compatibility
                self.diag_history: list[dict[str, object]] = []

        stubs = []
        for aid, rec in (self._areas or {}).items():
            stubs.append(_AreaStub(aid, rec.get("num_agents"), rec.get("personality_group_distribution")))
        stubs.sort(key=lambda a: a.unique_id)
        return stubs

    def _build_agent_stubs(self, *, pg_idx_by_agent_id: dict[int, int]):
        class _VoterStub:
            def __init__(self, *, agent_id: int, personality_group_idx: int,
                         personality_group: list[int]):
                self.unique_id = agent_id
                self.personality_group_idx = personality_group_idx
                self.personality_group = list(personality_group)
                self.personality = list(personality_group)
                self.assets = "-"

        stubs: dict[int, object] = {}
        n_groups = self.personality_groups.shape[0] if self.personality_groups.ndim == 2 else 0
        for aid in sorted(pg_idx_by_agent_id):
            pg_idx = pg_idx_by_agent_id[aid]
            if 0 <= pg_idx < n_groups:
                personality_group = self.personality_groups[pg_idx].tolist()
            else:
                personality_group = []
            stubs[aid] = _VoterStub(
                agent_id=aid,
                personality_group_idx=pg_idx,
                personality_group=personality_group,
            )
        return stubs

    # --- Properties expected by visualization elements ---
    @property
    def num_agents(self) -> int:
        return len(self.voting_agents)

    @property
    def num_personality_groups(self) -> int:
        return len(self.personality_groups)

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
        step = rec["step"]

        grid, grid_src_step = self.data.load_grid_with_source(step)
        self._apply_grid(grid)
        self.replay_recorded_step = step
        self.replay_grid_source_step = grid_src_step

        self.datacollector.add_model(rec["model"])

        self.datacollector.add_area_rows(step=step, areas=rec["areas"])

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

        h, w = arr.shape[0], arr.shape[1]

        if grid.width != w or grid.height != h:
            raise ValueError(f"Grid snapshot shape {(h, w)} does not match grid {(grid.height, grid.width)}.")

        flat = arr.T.ravel()  # (h,w) -> (w,h) x-major flatten

        for i, (cell, _pos) in enumerate(grid.coord_iter()):
            if cell is None:
                continue
            cell.color = flat[i]

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
