"""RunLogger.

Responsibilities:
- Write schema-v3 metadata files: `meta.yaml`, `static.json`
- Write schema-v3 parquet tables:
  - `steps.parquet`
  - `area_steps.parquet`
  - `agents.parquet`
  - `votes.parquet`

Timing semantics:
- Recorded step `t` is election-time state for step `t` (post election/reward,
  pre mutation of step `t`).
- Grid snapshots written by this logger are election-time snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import yaml
import hashlib
import json

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


@dataclass(frozen=True)
class RunContext:
    out_dir: Path
    run_seed: int
    rule_idx: int


@dataclass(frozen=True)
class _ConfigDumpParseResult:
    ok: bool
    payload: Dict[str, Any] | None
    error: str | None


class RunLogger:
    def __init__(
        self,
        out_dir: Path,
        run_seed: int,
        rule_idx: int,
        num_steps: int,
        store_grid: bool = True,
        compression: Optional[str] = "snappy",
    ) -> None:
        self.ctx = RunContext(out_dir=Path(out_dir),
                              run_seed=run_seed,
                              rule_idx=rule_idx)
        self.num_steps = int(num_steps)
        self.store_grid = bool(store_grid)
        self.compression = compression

        self.ctx.out_dir.mkdir(parents=True, exist_ok=True)

        self._steps_rows: List[Dict[str, Any]] = []
        self._area_steps_rows: List[Dict[str, Any]] = []
        self._agent_rows: List[Dict[str, Any]] = []
        self._votes_rows: List[Dict[str, Any]] = []
        self._current_step: Optional[int] = None
        self._num_colors: Optional[int] = None
        # Pre-mutation area snapshots keyed by (step, area_id)
        self._area_snapshots_by_step_area: Dict[tuple[int, int], Dict[str, Any]] = {}
        # Logger-local RNG for unbiased tie-breaks in logged top-k ranks.
        # Must be isolated from simulation RNG streams.
        seed = (run_seed * 1_000_003 + rule_idx * 9_173 + 17) % (2**63 - 1)
        self._vote_log_rng = np.random.default_rng(seed)

    # -----------------
    # Metadata
    # -----------------
    def write_meta(
        self,
        config: Any,
        *,
        config_ref: Optional[Path] = None,
        config_hash: Optional[str] = None,
        model: Optional[Model] = None,
    ) -> None:
        """Write meta.yaml with schema identifier and config reference."""
        if config_hash is None:
            cfg_dump = _load_required_config_dump(config)
            cfg_yaml = yaml.safe_dump(cfg_dump)
            config_hash = _hash_text(cfg_yaml)
        meta = {
            "schema": {
                "name": SCHEMA_NAME,
                "version": SCHEMA_VERSION,
                "step_indexing": STEP_INDEXING,
            },
            "run": {
                "run_seed": int(self.ctx.run_seed),
                "rule_idx": int(self.ctx.rule_idx),
                "rule_name": model.voting_rule_name if model is not None else None,
                "rule_impl_name": model.voting_rule_implementation_name if model is not None else None,
                "distance_idx": model.distance_idx if model is not None else None,
                "distance_name": model.distance_func_name if model is not None else None,
                "distance_impl_name": model.distance_func_implementation_name if model is not None else None,
                "quality_target_mode": model.quality_target_mode if model is not None else None,
                "puzzle_local_kappa": model.puzzle_local_kappa if model is not None else None,
                "puzzle_shock_prob": model.puzzle_shock_prob if model is not None else None,
            },
            "config_ref": str(config_ref) if config_ref is not None else None,
            "config_hash": config_hash,
        }
        with open(self.ctx.out_dir / "meta.yaml", "w") as f:
            yaml.safe_dump(meta, f)

    def write_static(self, model: Model) -> None:
        """Write static.json (schema v3 metadata) and static overlay artifacts."""

        height = int(model.height)
        width = int(model.width)
        num_colors = int(model.num_colors)
        num_areas = int(model.num_areas)
        num_agents = int(model.num_agents)
        self._num_colors = int(num_colors)

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
            "voting_rules": {
                "names": list(model.voting_rule_names),
                "impl_names": list(model.voting_rule_implementation_names),
                "selected_idx": int(self.ctx.rule_idx),
                "selected_name": model.voting_rule_name,
                "selected_impl_name": model.voting_rule_implementation_name,
            },
            "distance_functions": {
                "names": list(model.distance_func_names),
                "impl_names": list(model.distance_func_implementation_names),
                "selected_idx": int(model.distance_idx),
                "selected_name": model.distance_func_name,
                "selected_impl_name": model.distance_func_implementation_name,
            },
            "artifacts": {
                "steps": "steps.parquet",
                "area_steps": "area_steps.parquet",
                "agents": "agents.parquet",
                "votes": "votes.parquet",
                "cell_areas": "static_cell_areas.parquet",
                "cell_agents": "static_cell_agents.parquet",
            },
        }

        # Optional: personality_group metadata if present (useful for replay UI)
        raw_personality_groups = model.personality_groups
        global_pers_dist = model.personality_group_distribution
        # get personality_groups distributions per area
        area_distributions = {}
        areas = list(model.areas)
        for area in areas:
            area_infos = {}  # To save num_agents and personality_group_distribution
            a_id = area.unique_id
            num_agents = area.num_agents
            dist = area.personality_group_distribution
            area_infos["num_agents"] = num_agents
            area_infos["personality_group_distribution"] = _to_python(np.asarray(dist))
            area_distributions[str(a_id)] = area_infos
        payload = {
            "personality_groups": _to_python(np.asarray(raw_personality_groups)),
            "global_distribution": _to_python(global_pers_dist),
            "areas": area_distributions,
        }
        # Replay expects this key.
        static["personality_group_info"] = payload

        # Optional: per-agent static personal_opt_dist
        agents = list(model.voting_agents)
        if agents:
            pod: dict[str, list[float]] = {}
            for a in agents:
                if a is None:
                    continue
                pod[str(a.unique_id)] = _to_python(np.asarray(a.personal_opt_dist, dtype=np.float32))
            if pod:
                # static.json is a heterogeneous JSON payload; keep typing flexible here.
                static["personal_opt_dist"] = pod  # type: ignore[assignment]

        with open(self.ctx.out_dir / "static.json", "w") as f:
            json.dump(static, f, indent=2)

        # --- Typed static overlay artifacts for replay/analysis ---
        self._write_static_overlay_artifacts(model)

    def _write_static_overlay_artifacts(self, model: Model) -> None:
        """Write typed static cell/area/agent overlays used by replay and summaries."""
        cell_area_rows: list[dict[str, Any]] = []
        cell_agent_rows: list[dict[str, Any]] = []

        area_ids_by_agent: dict[int, set[int]] = {}
        for area in model.areas:
            area_id = int(area.unique_id)
            if area_id < 0:
                continue
            for agent in area.agents:
                agent_id = int(agent.unique_id)
                if agent_id < 0:
                    continue
                if agent_id not in area_ids_by_agent:
                    area_ids_by_agent[agent_id] = set()
                area_ids_by_agent[agent_id].add(area_id)

        for cell, (x, y) in model.grid.coord_iter():
            if cell is None:
                continue
            xi = int(x)
            yi = int(y)

            for area in cell.areas:
                area_id = int(area.unique_id)
                if area_id < 0:
                    continue
                cell_area_rows.append(
                    {
                        "x": np.int32(xi),
                        "y": np.int32(yi),
                        "area_id": np.int32(area_id),
                    }
                )

            for agent in cell.agents:
                agent_id = int(agent.unique_id)
                if agent_id < 0:
                    continue
                pg_idx = int(agent.personality_group_idx)
                area_ids = sorted(int(v) for v in area_ids_by_agent.get(agent_id, set()))
                if not area_ids:
                    # Some geometries intentionally leave cells outside all areas.
                    area_ids = [-1]
                for area_id in area_ids:
                    cell_agent_rows.append(
                        {
                            "x": np.int32(xi),
                            "y": np.int32(yi),
                            "area_id": np.int32(area_id),
                            "agent_id": np.int32(agent_id),
                            "personality_group_idx": np.int32(pg_idx),
                        }
                    )

        cell_areas_df = pd.DataFrame.from_records(cell_area_rows, columns=["x", "y", "area_id"])
        if not cell_areas_df.empty:
            cell_areas_df = cell_areas_df.drop_duplicates(subset=["x", "y", "area_id"], keep="first")
        cell_areas_df.to_parquet(self.ctx.out_dir / "static_cell_areas.parquet", engine="pyarrow", compression=self.compression)

        cell_agents_df = pd.DataFrame.from_records(
            cell_agent_rows,
            columns=["x", "y", "area_id", "agent_id", "personality_group_idx"],
        )
        if not cell_agents_df.empty:
            cell_agents_df = cell_agents_df.drop_duplicates(
                subset=["x", "y", "area_id", "agent_id"],
                keep="first",
            )
        cell_agents_df.to_parquet(self.ctx.out_dir / "static_cell_agents.parquet", engine="pyarrow", compression=self.compression)

    # -----------------
    # Logging
    # -----------------
    def attach_to_model(self, model: Model) -> None:
        """Attach output sinks to the model.

        - vote sink: used by `Area._tally_votes()` to emit participant vote rows
        - area snapshot sink: used by `Area._capture_area_snapshot_for_logger()`
        """
        if self._num_colors is None:
            self._num_colors = int(model.num_colors)
        model.register_output_sinks(
            vote_sink=self._on_vote,
            area_snapshot_sink=self._on_area_snapshot,
        )

    def detach_from_model(self, model: Model) -> None:
        """Detach output sinks from the model."""
        model.clear_output_sinks()

    def log_step(self, step: int, model: Model, grid_snapshot: Optional[np.ndarray] = None) -> None:
        """Append schema-v3 rows for this step.

        Args:
            step: Recorded step number (schema v3 is 1-based).
            model: The ParticipationModel (pre-mutation state).
            grid_snapshot: Optional HxW array to write to grids/ (1-based, pre-mutation).
        Notes:
            steps.parquet and area_steps.parquet color distributions are derived
            from pre-mutation snapshots captured during the election.
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

    def _on_area_snapshot(self, *, area: Area, snapshot: Dict[str, Any]) -> None:
        """Capture pre-mutation area snapshot for steps/area_steps.parquet."""
        if self._current_step is None:
            return
        area_id = int(area.unique_id)
        snapshot_copy = dict(snapshot)
        for key in ("area_color", "elected_color"):
            val = snapshot_copy.get(key)
            if isinstance(val, np.ndarray):
                snapshot_copy[key] = val.copy()
            elif isinstance(val, list):
                snapshot_copy[key] = list(val)
        self._area_snapshots_by_step_area[(int(self._current_step), area_id)] = snapshot_copy

    def finalize(self) -> None:
        """Write Parquet artifacts (steps/area_steps/agents/votes)."""
        steps_df = pd.DataFrame(self._steps_rows)
        area_steps_df = pd.DataFrame(self._area_steps_rows)
        agents_df = pd.DataFrame(self._agent_rows)

        votes_df = pd.DataFrame(self._votes_rows)
        if votes_df.empty:
            votes_df = self._empty_votes_df()
        elif "voted_altruistically" in votes_df.columns:
            # Keep explicit tri-state boolean semantics (True/False/<NA>).
            votes_df["voted_altruistically"] = votes_df["voted_altruistically"].astype("boolean")

        # Validate before writing (helps fail fast during development)
        validate_steps_df(steps_df)
        validate_area_steps_df(area_steps_df)
        validate_agents_df(agents_df)
        validate_votes_df(votes_df)

        steps_df.to_parquet(self.ctx.out_dir / "steps.parquet", engine="pyarrow", compression=self.compression)
        area_steps_df.to_parquet(self.ctx.out_dir / "area_steps.parquet", engine="pyarrow", compression=self.compression)
        agents_df.to_parquet(self.ctx.out_dir / "agents.parquet", engine="pyarrow", compression=self.compression)
        votes_df.to_parquet(self.ctx.out_dir / "votes.parquet", engine="pyarrow", compression=self.compression)

    def _empty_votes_df(self) -> pd.DataFrame:
        """Build a schema-valid empty votes frame with explicit dtypes."""
        num_colors = int(self._num_colors) if self._num_colors is not None else 0
        if num_colors <= 0:
            raise RuntimeError("Cannot build empty votes.parquet schema: num_colors is not initialized.")

        cols: Dict[str, pd.Series] = {
            "run_seed": pd.Series(dtype="int32"),
            "rule_idx": pd.Series(dtype="int16"),
            "step": pd.Series(dtype="int32"),
            "area_id": pd.Series(dtype="int32"),
            "agent_id": pd.Series(dtype="int32"),
            "participating": pd.Series(dtype="boolean"),
            "confidence": pd.Series(dtype="float32"),
            "voted_altruistically": pd.Series(dtype="boolean"),
            "rank_1_option_id": pd.Series(dtype="Int32"),
            "rank_1_oppose_score": pd.Series(dtype="float32"),
            "rank_2_option_id": pd.Series(dtype="Int32"),
            "rank_2_oppose_score": pd.Series(dtype="float32"),
            "rank_3_option_id": pd.Series(dtype="Int32"),
            "rank_3_oppose_score": pd.Series(dtype="float32"),
        }
        for i in range(num_colors):
            cols[f"estim_dst_color_{i}"] = pd.Series(dtype="float32")
        return pd.DataFrame(cols)

    # -----------------
    # Extraction helpers
    # -----------------
    def _extract_steps_row(self, step: int, model: Model) -> Dict[str, Any]:
        snap = model.step_metrics_snapshot
        if not isinstance(snap, dict):
            raise RuntimeError(f"Missing step_metrics_snapshot for step {step}.")
        required = (
            "collective_assets",
            "gini_index",
            "turnout",
            "mean_altruism",
            "mean_dissatisfaction",
        )
        missing = [k for k in required if k not in snap]
        if missing:
            raise RuntimeError(f"Missing step_metrics_snapshot fields for step {step}: {missing}")

        row: Dict[str, Any] = {
            "run_seed": np.int32(self.ctx.run_seed),
            "rule_idx": np.int16(self.ctx.rule_idx),
            "step": np.int32(step),
            "collective_assets": float(snap["collective_assets"]),
            "gini_index": np.int16(int(snap["gini_index"])),
            "turnout": float(snap["turnout"]),
            "mean_altruism": float(snap["mean_altruism"]),
            "mean_dissatisfaction": float(snap["mean_dissatisfaction"]),
        }

        pre_colors = self._get_pre_mutation_global_colors(step=step, model=model)

        if pre_colors is not None:
            for i, v in enumerate(pre_colors):
                row[f"color_{i}"] = float(v)

        return row

    def _get_pre_mutation_global_colors(self, *, step: int, model: Model) -> Optional[np.ndarray]:
        # Authoritative source for global color_* in steps.parquet:
        # model.global_color_dst at election-time state.
        num_colors = int(model.num_colors)
        if num_colors <= 0:
            return None

        vals = np.asarray(model.global_color_dst, dtype=np.float64)
        if vals.ndim != 1 or vals.size != num_colors:
            raise RuntimeError(
                f"Invalid model.global_color_dst shape at step {step}: expected ({num_colors},), got {vals.shape}."
            )
        if not np.all(np.isfinite(vals)):
            raise RuntimeError(f"Invalid model.global_color_dst values at step {step}: non-finite entries found.")

        total = float(np.sum(vals))
        if total <= 0.0:
            raise RuntimeError(f"Invalid model.global_color_dst values at step {step}: sum must be > 0.")

        # Normalize to absorb tiny floating-point drift.
        return (vals / total).astype(np.float32)

    def _extract_area_steps_rows(self, step: int, model: Model) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        areas = model.areas
        num_colors = int(model.num_colors)

        def _apply_election_vectors(
            *,
            r_dict,
            elected_color_vec,
            area_color_vec,
            puzzle_color_vec=None,
            group_outcome_distance_vec=None,
        ) -> None:
            """Fill expanded vector columns + winning_option_id into r dictionary.

            `elected_color_vec` is a length-C ordering (ints).
            `area_color_vec` is a length-C distribution (floats).
            """
            if elected_color_vec is not None:
                vo = np.asarray(elected_color_vec, dtype=np.int16).tolist()
                for i in range(num_colors):
                    r_dict[f"elected_color_{i}"] = np.int16(vo[i])
                try:
                    oid = int(model.option_id_for_ordering(elected_color_vec))
                    if oid >= 0:
                        r_dict["winning_option_id"] = np.int32(oid)
                except (AttributeError, ValueError, TypeError):
                    pass

            if area_color_vec is not None:
                cdv = np.asarray(area_color_vec, dtype=np.float32)
                for i in range(num_colors):
                    r_dict[f"area_color_{i}"] = np.float32(cdv[i])
            if puzzle_color_vec is not None:
                pdv = np.asarray(puzzle_color_vec, dtype=np.float32)
                for i in range(num_colors):
                    r_dict[f"puzzle_color_{i}"] = np.float32(pdv[i])
            if group_outcome_distance_vec is not None:
                godv = np.asarray(group_outcome_distance_vec, dtype=np.float32)
                for i in range(godv.size):
                    r_dict[f"group_outcome_distance_{i}"] = np.float32(godv[i])

        for area in areas:
            area_id = area.unique_id

            # Base row
            r: Dict[str, Any] = {
                "run_seed": np.int32(self.ctx.run_seed),
                "rule_idx": np.int16(self.ctx.rule_idx),
                "step": np.int32(step),
                "area_id": np.int32(area_id),
                "eligible_voters": np.int32(area.num_eligible_voters_last),
                # participants is overwritten from the pre-mutation area snapshot below.
                "participants": np.int32(0),
                "turnout": np.float32(area.voter_turnout),  # In percent
                "fee_pool": np.float32(area.election_fee_pool),
                "winning_option_id": np.int32(-1),
                "grid_ordering_id": np.int32(-1),
                "puzzle_ordering_id": np.int32(-1),
                "dist_to_reality": np.float32(area.dist_to_reality),
                "puzzle_distance": np.float32(area.puzzle_distance),
                "gini_index": np.int16(0),
            }

            snapshot = self._area_snapshots_by_step_area.pop((int(step), area_id), None)
            if snapshot is None:
                raise RuntimeError(
                    f"Missing pre-mutation area snapshot for step {step}, area {area_id}."
                )
            required = (
                "eligible_voters",
                "participants",
                "turnout",
                "fee_pool",
                "dist_to_reality",
                "puzzle_distance",
                "gini_index",
                "area_color",
                "elected_color",
                "grid_ordering_id",
                "puzzle_ordering_id",
            )
            missing = [k for k in required if k not in snapshot]
            if missing:
                raise RuntimeError(
                    f"Missing required pre-mutation snapshot fields for step {step}, "
                    f"area {area_id}: {missing}"
                )

            r["eligible_voters"] = np.int32(int(snapshot["eligible_voters"]))
            r["participants"] = np.int32(int(snapshot["participants"]))
            r["turnout"] = np.float32(float(snapshot["turnout"]))
            r["fee_pool"] = np.float32(float(snapshot["fee_pool"]))
            r["dist_to_reality"] = np.float32(float(snapshot["dist_to_reality"]))
            r["puzzle_distance"] = np.float32(float(snapshot["puzzle_distance"]))
            r["gini_index"] = np.int16(int(snapshot["gini_index"]))
            r["grid_ordering_id"] = np.int32(int(snapshot["grid_ordering_id"]))
            r["puzzle_ordering_id"] = np.int32(int(snapshot["puzzle_ordering_id"]))
            voted_ordering = snapshot.get("elected_color", None)
            cd = snapshot.get("area_color", None)
            puzzle_cd = snapshot.get("puzzle_color", None)
            group_outcome_distance = snapshot.get("group_outcome_distance", None)
            if group_outcome_distance is None:
                group_outcome_distance = (
                    area.group_outcome_distance
                    or [float("nan")] * int(model.num_personality_groups)
                )
            if cd is None:
                raise RuntimeError(
                    f"Missing pre-mutation area_color for step {step}, area {area_id}."
                )

            _apply_election_vectors(
                r_dict=r,
                elected_color_vec=voted_ordering,
                area_color_vec=cd,
                puzzle_color_vec=puzzle_cd,
                group_outcome_distance_vec=group_outcome_distance,
            )

            rows.append(r)

        return rows

    def _extract_agent_rows(self, step: int, model: Model) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        agents = model.voting_agents
        for a in agents:
            rows.append(
                {
                    "run_seed": np.int32(self.ctx.run_seed),
                    "rule_idx": np.int16(self.ctx.rule_idx),
                    "step": np.int32(step),
                    "agent_id": np.int32(int(a.unique_id)),
                    "assets": np.float32(float(a.assets)),
                    "num_elections_participated": np.int32(int(a.num_elections_participated)),
                    "personality_group_idx": np.int16(a.personality_group_idx),
                    "eligible_for_election": bool(a.eligible_for_election),
                    "participating": bool(a.participating),
                    "election_fee": np.float32(float(a.election_fee)),
                    "reward_personal": np.float32(float(a.reward_personal)),
                    "election_delta_abs": np.float32(float(a.election_delta_abs)),
                    "election_delta_rel": np.float32(float(a.election_delta_rel)),
                    "participation_baseline": np.float32(float(a.participation_baseline)),
                    "participation_signal": np.float32(float(a.participation_signal)),
                    "participation_signal_group_component": np.float32(float(a.participation_signal_group_component)),
                    "participation_signal_fee_component": np.float32(float(a.participation_signal_fee_component)),
                    "q_participation": np.float32(float(a.q_participation)),
                    "participation_probability": np.float32(float(a.participation_probability())),
                    "altruism_factor": np.float32(float(a.altruism_factor)),
                    "dissatisfaction_value": np.float32(float(a.dissatisfaction_value)),
                    "dissatisfaction_baseline": np.float32(float(a.dissatisfaction_baseline)),
                    "dissatisfaction_signal": np.float32(float(a.dissatisfaction_signal)),
                }
            )
        return rows

    def _on_vote(
        self,
        *,
        area: Area,
        agent: VoteAgent,
        oppose_scores: np.ndarray,
        est_dist: np.ndarray,
        confidence: float,
        voted_altruistically: bool | None = None,
    ) -> None:
        """Receive one participant vote and append a schema v3 vote row."""
        if self._current_step is None:
            return
        step = int(self._current_step)
        area_id = area.unique_id
        agent_id = agent.unique_id
        participating = agent.participating

        scores = np.asarray(oppose_scores, dtype=np.float32)
        if scores.ndim != 1:
            return

        # Pick the 3 best (lowest oppose score) options.
        # Tie-break must be unbiased and must not consume simulation RNG.
        rand = self._vote_log_rng.random(scores.shape[0]).astype(np.float32)
        order = np.lexsort((rand, scores))
        top = order[:3].tolist()

        row: Dict[str, Any] = {
            "run_seed": np.int32(self.ctx.run_seed),
            "rule_idx": np.int16(self.ctx.rule_idx),
            "step": np.int32(step),
            "area_id": np.int32(area_id),
            "agent_id": np.int32(agent_id),
            "participating": bool(participating),
            "confidence": np.float32(0.0 if confidence is None else float(confidence)),
            "voted_altruistically": (
                bool(voted_altruistically) if isinstance(voted_altruistically, bool) else pd.NA
            ),
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
        missing_or_invalid_estimate = (
            dist is None
            or dist.ndim != 1
            or (dist.size > 0 and np.all(dist == 0.0))
        )
        if missing_or_invalid_estimate:
            # Emit NaNs to avoid masking missing estimate_real_distribution()
            # (all-zero vectors are treated as invalid for this context because
            # they cannot represent a valid probability distribution).
            num_colors = int(agent.model.num_colors)
            dist = np.full(num_colors, np.nan, dtype=np.float32)
        for i in range(dist.shape[0]):
            row[f"estim_dst_color_{i}"] = np.float32(dist[i])

        self._votes_rows.append(row)

    def _grid_filename(self, step: int) -> Path:
        pad = len(str(int(self.num_steps))) if self.num_steps is not None else 3
        return self.ctx.out_dir / "grids" / f"grid_{int(step):0{pad}d}.npy"

    def _write_grid_snapshot(self, *, step: int, grid_snapshot: np.ndarray) -> None:
        grids_dir = self.ctx.out_dir / "grids"
        grids_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(self._grid_filename(step)), grid_snapshot)


# Backward-compatible aliases for existing imports during migration.
RunContextV2 = RunContext
RunLoggerV2 = RunLogger


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


def _parse_config_dump(config: Any) -> _ConfigDumpParseResult:
    # Pydantic v2
    if hasattr(config, "model_dump"):
        try:
            dumped = config.model_dump(mode="json")
        except (TypeError, ValueError) as e:
            return _ConfigDumpParseResult(
                ok=False,
                payload=None,
                error=f"Failed to dump config via model_dump(mode='json'): {e}",
            )
        if not isinstance(dumped, dict):
            return _ConfigDumpParseResult(
                ok=False,
                payload=None,
                error=f"Invalid config dump type from model_dump: {type(dumped).__name__}",
            )
        return _ConfigDumpParseResult(ok=True, payload=dict(dumped), error=None)
    # Pydantic v1
    if hasattr(config, "dict"):
        try:
            dumped = json.loads(config.json())
        except (TypeError, ValueError, json.JSONDecodeError) as e:
            return _ConfigDumpParseResult(
                ok=False,
                payload=None,
                error=f"Failed to dump config via config.json(): {e}",
            )
        if not isinstance(dumped, dict):
            return _ConfigDumpParseResult(
                ok=False,
                payload=None,
                error=f"Invalid config dump type from config.json(): {type(dumped).__name__}",
            )
        return _ConfigDumpParseResult(ok=True, payload=dict(dumped), error=None)
    if isinstance(config, dict):
        return _ConfigDumpParseResult(ok=True, payload=dict(config), error=None)
    return _ConfigDumpParseResult(
        ok=False,
        payload=None,
        error=f"Unsupported config type for meta hash dump: {type(config).__name__}",
    )


def _load_required_config_dump(config: Any) -> Dict[str, Any]:
    parsed = _parse_config_dump(config)
    if not parsed.ok or parsed.payload is None:
        raise RuntimeError(parsed.error or "Failed to dump config for meta hash")
    return parsed.payload


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
