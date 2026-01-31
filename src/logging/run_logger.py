"""RunLoggerV2

Phase: schema v2 migration (Batch 1.0)

Responsibilities (Batch 1.0 only):
- Write schema v2 run metadata files: meta.yaml, static.json
- Write Parquet tables with real rows:
  - steps.parquet
  - area_steps.parquet
  - agents.parquet

Not implemented yet (future batches):
- votes.parquet
- pre-mutation snapshot hooks

This module intentionally keeps core model logic unchanged and reads values from the
model/areas/agents after each model.step().
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import yaml

from src.logging.output_schema import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    STEP_INDEXING,
    validate_steps_df,
    validate_area_steps_df,
    validate_agents_df,
    validate_votes_df,
)
from src.agents.area import Area
from src.agents.vote_agent import VoteAgent
from src.models.participation_model import ParticipationModel as Model
from src.utils.metrics import (
    get_area_border_grid,
    get_agents_per_cell_grid,
    get_agent_strings_per_cell_grid,
    get_area_strings_per_cell_grid,
)


@dataclass(frozen=True)
class RunContextV2:
    out_dir: Path
    run_seed: int
    rule_idx: int


class RunLoggerV2:
    def __init__(
        self,
        out_dir: Path,
        run_seed: int,
        rule_idx: int,
        num_steps: int,
        store_grid: bool = True,
        compression: Optional[str] = "snappy",
    ) -> None:
        self.ctx = RunContextV2(out_dir=Path(out_dir), run_seed=int(run_seed), rule_idx=int(rule_idx))
        self.num_steps = int(num_steps)
        self.store_grid = bool(store_grid)
        self.compression = compression

        self.ctx.out_dir.mkdir(parents=True, exist_ok=True)

        self._steps_rows: List[Dict[str, Any]] = []
        self._area_steps_rows: List[Dict[str, Any]] = []
        self._agent_rows: List[Dict[str, Any]] = []
        self._votes_rows: List[Dict[str, Any]] = []
        self._current_step: Optional[int] = None
        # Pre-mutation area snapshots emitted from Area.step() (Batch 3)
        self._area_snapshots_by_step_area: Dict[tuple[int, int], Dict[str, Any]] = {}

    # -----------------
    # Metadata
    # -----------------
    def write_meta(self, config: Any) -> None:
        """Write meta.yaml with schema identifier and config dump."""
        meta = {
            "schema": {
                "name": SCHEMA_NAME,
                "version": SCHEMA_VERSION,
                "step_indexing": STEP_INDEXING,
            },
            "run": {
                "run_seed": int(self.ctx.run_seed),
                "rule_idx": int(self.ctx.rule_idx),
            },
            "config": _safe_config_dump(config),
        }
        with open(self.ctx.out_dir / "meta.yaml", "w") as f:
            yaml.safe_dump(meta, f)

    def write_static(self, model: Model) -> None:
        """Write static.json (schema v2 metadata) and static overlay artifacts."""

        height = int(getattr(model, "height", 0) or 0)
        width = int(getattr(model, "width", 0) or 0)
        num_colors = int(getattr(model, "num_colors", 0) or 0)
        num_areas = int(getattr(model, "num_areas", 0) or 0)
        num_agents = int(getattr(model, "num_agents", 0) or 0)

        static = {
            "schema": {
                "name": SCHEMA_NAME,
                "version": SCHEMA_VERSION,
            },
            "height": height,
            "width": width,
            "num_colors": num_colors,
            "num_areas": num_areas,
            "num_agents": num_agents,
            "step_indexing": {
                "meaning": STEP_INDEXING,
                "first_recorded_step": 1,
                "grid_file": _grid_pattern(self.num_steps),
            },
            "artifacts": {
                "steps": "steps.parquet",
                "area_steps": "area_steps.parquet",
                "agents": "agents.parquet",
                "votes": "votes.parquet",
                "area_borders": "area_borders.npy",
                "agents_per_cell": "agents_per_cell.npy",
                "agent_strings_per_cell": "agent_strings_per_cell.npy",
                "cell_areas": "area_strings_per_cell.npy",
            },
        }

        # Optional: personality metadata if present (useful for replay UI)
        raw_personalities = getattr(model, "personalities", None)
        global_pers_dist = getattr(model, "personality_distribution", None)
        if raw_personalities is not None and global_pers_dist is not None:
            static["personality_info"] = {
                "personalities": _to_python(np.asarray(raw_personalities)),
                "global_distribution": _to_python(global_pers_dist),
            }

        # Optional: per-agent static personal_opt_dist
        agents = list(model.voting_agents)
        if agents:
            pod: dict[str, list[float]] = {}
            for a in agents:
                if a is None:
                    continue
                dist = getattr(a, "personal_opt_dist", None)
                if dist is None:
                    continue
                pod[str(a.unique_id)] = _to_python(np.asarray(dist, dtype=np.float32))
            if pod:
                # static.json is a heterogeneous JSON payload; keep typing flexible here.
                static["personal_opt_dist"] = pod  # type: ignore[assignment]

        import json

        with open(self.ctx.out_dir / "static.json", "w") as f:
            json.dump(static, f, indent=2)

        # --- Static overlay artifacts for replay ---
        borders = get_area_border_grid(model)
        np.save(str(self.ctx.out_dir / "area_borders.npy"), np.asarray(borders, dtype=bool))

        apc = get_agents_per_cell_grid(model)
        np.save(str(self.ctx.out_dir / "agents_per_cell.npy"), np.asarray(apc, dtype=np.int32))

        as_pc = get_agent_strings_per_cell_grid(model)
        np.save(str(self.ctx.out_dir / "agent_strings_per_cell.npy"), np.asarray(as_pc, dtype=str))

        area_strs = get_area_strings_per_cell_grid(model)
        np.save(str(self.ctx.out_dir / "area_strings_per_cell.npy"), np.asarray(area_strs, dtype=str))

    # -----------------
    # Logging
    # -----------------
    def attach_to_model(self, model: Model) -> None:
        """Attach schema-v2 sinks to the model.

        - vote sink: used by Area._tally_votes() to emit participant vote rows.
        - area snapshot sink: used by Area.step() to emit a post-election/pre-mutation
          snapshot for area_steps.parquet.
        """
        setattr(model, "_schema_v2_vote_sink", self._on_vote)
        setattr(model, "_schema_v2_area_snapshot_sink", self._on_area_snapshot)

    def detach_from_model(self, model: Model) -> None:
        """Detach schema-v2 sinks from the model."""
        if getattr(model, "_schema_v2_vote_sink", None) is self._on_vote:
            delattr(model, "_schema_v2_vote_sink")
        if getattr(model, "_schema_v2_area_snapshot_sink", None) is self._on_area_snapshot:
            delattr(model, "_schema_v2_area_snapshot_sink")

    def log_step(self, step: int, model: Model, grid_snapshot: Optional[np.ndarray] = None) -> None:
        """Append schema-v2 rows for this step.

        Args:
            step: Recorded step number (schema v2 is 1-based).
            model: The ParticipationModel.
            grid_snapshot: Optional HxW array to write to grids/ (1-based).
        """
        s = int(step)
        self._current_step = s
        self._steps_rows.append(self._extract_steps_row(s, model))
        self._area_steps_rows.extend(self._extract_area_steps_rows(s, model))
        self._agent_rows.extend(self._extract_agent_rows(s, model))

        if self.store_grid and grid_snapshot is not None:
            self._write_grid_snapshot(step=s, grid_snapshot=np.asarray(grid_snapshot))

    def begin_step(self, step: int) -> None:
        """Set the current step used by sink callbacks."""
        self._current_step = int(step)

    def end_step(self) -> None:
        """Clear current step after finishing a model step."""
        self._current_step = None

    def finalize(self) -> None:
        """Write Parquet artifacts (steps/area_steps/agents/votes)."""
        steps_df = pd.DataFrame(self._steps_rows)
        area_steps_df = pd.DataFrame(self._area_steps_rows)
        agents_df = pd.DataFrame(self._agent_rows)

        votes_df = pd.DataFrame(self._votes_rows)
        if votes_df.empty:
            # Create an empty frame with required columns so schema validation passes
            votes_df = pd.DataFrame(
                columns=[
                    "run_seed",
                    "rule_idx",
                    "step",
                    "area_id",
                    "agent_id",
                    "participated",
                    "confidence",
                    "rank_1_option_id",
                    "rank_1_oppose_score",
                    "rank_2_option_id",
                    "rank_2_oppose_score",
                    "rank_3_option_id",
                    "rank_3_oppose_score",
                ]
            )

        # Validate before writing (helps fail fast during development)
        validate_steps_df(steps_df)
        validate_area_steps_df(area_steps_df)
        validate_agents_df(agents_df)
        validate_votes_df(votes_df)

        steps_df.to_parquet(self.ctx.out_dir / "steps.parquet", engine="pyarrow", compression=self.compression)
        area_steps_df.to_parquet(self.ctx.out_dir / "area_steps.parquet", engine="pyarrow", compression=self.compression)
        agents_df.to_parquet(self.ctx.out_dir / "agents.parquet", engine="pyarrow", compression=self.compression)
        votes_df.to_parquet(self.ctx.out_dir / "votes.parquet", engine="pyarrow", compression=self.compression)

    # -----------------
    # Extraction helpers
    # -----------------
    def _extract_steps_row(self, step: int, model: Model) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "run_seed": np.int32(self.ctx.run_seed),
            "rule_idx": np.int16(self.ctx.rule_idx),
            "step": np.int32(step),
            "collective_assets": np.int64(0),
            "gini_index": np.int16(0),
            "turnout": np.float32(0.0),
        }

        dc = getattr(model, "datacollector", None)
        if dc is None:
            return row

        df = dc.get_model_vars_dataframe()
        if df is None or len(df) == 0:
            return row

        last = df.iloc[-1].to_dict()

        # Prefer snake_case (live + replay v2 use this)
        if "collective_assets" in last:
            row["collective_assets"] = np.int64(last["collective_assets"])
        elif "Collective assets" in last:
            row["collective_assets"] = np.int64(last["Collective assets"])

        if "gini_index" in last:
            row["gini_index"] = np.int16(last["gini_index"])
        elif "Gini Index (0-100)" in last:
            row["gini_index"] = np.int16(last["Gini Index (0-100)"])

        if "turnout" in last:
            row["turnout"] = np.float32(last["turnout"])
        elif "Voter turnout globally (in percent)" in last:
            row["turnout"] = np.float32(last["Voter turnout globally (in percent)"])

        # Optional per-color series: snake_case color_0...color_{C-1} (preferred)
        for k, v in last.items():
            if isinstance(k, str) and k.startswith("color_"):
                suf = k.split("_", 1)[1]
                if suf.isdigit():
                    row[k] = np.float32(v)

        # Legacy fallback: "Color 0"..."Color {C-1}"
        for k, v in last.items():
            if isinstance(k, str) and k.startswith("Color "):
                parts = k.split(" ")
                if len(parts) == 2 and parts[1].isdigit():
                    idx = int(parts[1])
                    row[f"color_{idx}"] = np.float32(v)

        return row

    def _extract_area_steps_rows(self, step: int, model: Model) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        areas = list(getattr(model, "areas", []) or [])
        num_colors = int(getattr(model, "num_colors", 0) or 0)

        options = getattr(model, "options", None)
        if options is not None:
            options = np.asarray(options)

        def _apply_election_vectors(*, r_dict, elected_color_vec, area_color_vec) -> None:
            """Fill expanded vector columns + winning_option_id into r dictionary.

            `elected_color_vec` is a length-C ordering (ints).
            `area_color_vec` is a length-C distribution (floats).
            """
            if elected_color_vec is not None:
                vo = np.asarray(elected_color_vec, dtype=np.int16).tolist()
                for i in range(num_colors):
                    r_dict[f"elected_color_{i}"] = np.int16(vo[i])
                if options is not None:
                    try:
                        matches = np.nonzero((options == np.asarray(elected_color_vec)).all(axis=1))[0]
                        if len(matches) > 0:
                            r_dict["winning_option_id"] = np.int32(int(matches[0]))
                    except (ValueError, IndexError, TypeError):
                        pass

            if area_color_vec is not None:
                cdv = np.asarray(area_color_vec, dtype=np.float32)
                for i in range(num_colors):
                    r_dict[f"area_color_{i}"] = np.float32(cdv[i])

        for area in areas:
            if area is None:
                continue
            area_id = int(getattr(area, "unique_id", -1))

            snap = self._area_snapshots_by_step_area.get((step, area_id))

            # Base row
            r: Dict[str, Any] = {
                "run_seed": np.int32(self.ctx.run_seed),
                "rule_idx": np.int16(self.ctx.rule_idx),
                "step": np.int32(step),
                "area_id": np.int32(area_id),
                "eligible_voters": np.int32(area.num_agents),
                # Not tracked explicitly yet; default 0.
                "participants": np.int32(0),
                "turnout": np.float32(area.voter_turnout / 100.0
                    if area.voter_turnout > 1.0 else float(area.voter_turnout)
                ),
                "election_cost_rate": np.float32(float(getattr(model, "election_costs", 0.0) or 0.0)),
                "fee_pool": np.float32(float(getattr(area, "_election_fee_pool", 0.0) or 0.0)),
                "winning_option_id": np.int32(-1),
                "dist_to_reality": np.float32(float(getattr(area, "dist_to_reality", 0.0) or 0.0)),
                "gini_index": np.int16(0),
            }

            if snap is not None:
                # Prefer the pre-mutation snapshot (Batch 3).
                # TODO(schema-v2): remove post-step fallback once snapshot coverage is guaranteed.
                if "eligible_voters" in snap and snap["eligible_voters"] is not None:
                    r["eligible_voters"] = np.int32(int(snap["eligible_voters"]))
                if "participants" in snap and snap["participants"] is not None:
                    r["participants"] = np.int32(int(snap["participants"]))
                if "turnout" in snap and snap["turnout"] is not None:
                    tv = float(snap["turnout"])
                    r["turnout"] = np.float32(tv / 100.0 if tv > 1.0 else tv)
                if "election_cost_rate" in snap and snap["election_cost_rate"] is not None:
                    r["election_cost_rate"] = np.float32(float(snap["election_cost_rate"]))
                if "fee_pool" in snap and snap["fee_pool"] is not None:
                    r["fee_pool"] = np.float32(float(snap["fee_pool"]))
                if "dist_to_reality" in snap and snap["dist_to_reality"] is not None:
                    r["dist_to_reality"] = np.float32(float(snap["dist_to_reality"]))

                elected_color = snap.get("elected_color")
                area_color = snap.get("area_color")
                _apply_election_vectors(r_dict=r,
                                        elected_color_vec=elected_color,
                                        area_color_vec=area_color)
            else:
                # Fallback: post-step reads (may be post-mutation). Kept for safety.
                # TODO(schema-v2): remove fallback once snapshot hook is tested across configs.
                voted_ordering = getattr(area, "voted_ordering", None)
                cd = getattr(area, "color_distribution", None)
                _apply_election_vectors(r_dict=r,
                                        elected_color_vec=voted_ordering,
                                        area_color_vec=cd)

            # area gini from agents' assets (same as old replay logger)
            agents = list(getattr(area, "agents", []) or [])
            if agents:
                assets = [float(getattr(a, "assets", 0.0) or 0.0) for a in agents]
                # reuse metric helper indirectly: gini_index_0_100 exists in utils.metrics
                from src.utils.metrics import gini_index_0_100

                r["gini_index"] = np.int16(int(gini_index_0_100(assets)))

            rows.append(r)

        return rows

    def _extract_agent_rows(self, step: int, model: Model) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        agents = list(getattr(model, "voting_agents", []) or [])
        for a in agents:
            if a is None:
                continue
            rows.append(
                {
                    "run_seed": np.int32(self.ctx.run_seed),
                    "rule_idx": np.int16(self.ctx.rule_idx),
                    "step": np.int32(step),
                    "agent_id": np.int32(int(getattr(a, "unique_id", -1))),
                    "row": np.int16(int(getattr(a, "row", 0) or 0)),
                    "col": np.int16(int(getattr(a, "col", 0) or 0)),
                    "assets": np.float32(float(getattr(a, "assets", 0.0) or 0.0)),
                    "num_elections_participated": np.int32(int(getattr(a, "num_elections_participated", 0) or 0)),
                    "personality_idx": np.int16(int(getattr(a, "personality_idx", -1) or -1)),
                }
            )
        return rows

    def _on_vote(self, *, area: Area, agent: VoteAgent, oppose_scores: np.ndarray,
        est_dist: np.ndarray, confidence: float) -> None:
        """Receive one participant vote and append a schema v2 vote row."""
        if self._current_step is None:
            return
        step = int(self._current_step)
        area_id = area.unique_id
        agent_id = agent.unique_id

        scores = np.asarray(oppose_scores, dtype=np.float32)
        if scores.ndim != 1:
            return

        # Pick the 3 best (lowest oppose score) options.
        # For ties, use a stable secondary sort by option id for determinism.
        order = np.lexsort((agent.model.option_vec, scores))
        top = order[:3].tolist()

        row: Dict[str, Any] = {
            "run_seed": np.int32(self.ctx.run_seed),
            "rule_idx": np.int16(self.ctx.rule_idx),
            "step": np.int32(step),
            "area_id": np.int32(area_id),
            "agent_id": np.int32(agent_id),
            "participated": True,
            "confidence": np.float32(0.0 if confidence is None else float(confidence)),
            "rank_1_option_id": pd.NA,
            "rank_1_oppose_score": np.float32(np.nan),
            "rank_2_option_id": pd.NA,
            "rank_2_oppose_score": np.float32(np.nan),
            "rank_3_option_id": pd.NA,
            "rank_3_oppose_score": np.float32(np.nan),
        }

        for i, opt in enumerate(top, start=1):
            row[f"rank_{i}_option_id"] = np.int32(int(opt))
            row[f"rank_{i}_oppose_score"] = np.float32(float(scores[int(opt)]))

        # estim_dst_color_* expanded columns
        dist = np.asarray(est_dist, dtype=np.float32) if est_dist is not None else None
        if dist is None or dist.ndim != 1:
            # Emit zeros to satisfy contract (will be refined later if needed)
            num_colors = int(agent.model.num_colors)
            dist = np.zeros(num_colors, dtype=np.float32)
        for i in range(dist.shape[0]):
            row[f"estim_dst_color_{i}"] = np.float32(dist[i])

        self._votes_rows.append(row)

    def _on_area_snapshot(self, *, area: Area, snapshot: Dict[str, Any]) -> None:
        """Receive a post-election/pre-mutation snapshot for one area."""
        if self._current_step is None:
            return
        step = int(self._current_step)
        area_id = int(getattr(area, "unique_id", -1))
        self._area_snapshots_by_step_area[(step, area_id)] = dict(snapshot)

    def _grid_filename(self, step: int) -> Path:
        pad = len(str(int(self.num_steps))) if self.num_steps is not None else 3
        return self.ctx.out_dir / "grids" / f"grid_{int(step):0{pad}d}.npy"

    def _write_grid_snapshot(self, *, step: int, grid_snapshot: np.ndarray) -> None:
        grids_dir = self.ctx.out_dir / "grids"
        grids_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(self._grid_filename(step)), grid_snapshot)


def _grid_pattern(num_steps: int) -> str:
    pad = len(str(int(num_steps))) if num_steps is not None else 3
    return f"grid_%0{pad}d.npy"


def _to_python(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return {}


def _safe_config_dump(config: Any) -> Dict[str, Any]:
    # Pydantic v2
    if hasattr(config, "model_dump"):
        try:
            return config.model_dump()
        except (TypeError, ValueError):
            return {}
    # Pydantic v1
    if hasattr(config, "dict"):
        try:
            return config.dict()
        except (TypeError, ValueError):
            return {}
    return {}
