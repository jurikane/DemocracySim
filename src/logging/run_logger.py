"""RunLoggerV2.

Responsibilities:
- Write schema-v2 metadata files: `meta.yaml`, `static.json`
- Write schema-v2 parquet tables:
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
        self._num_colors: Optional[int] = None
        # Pre-mutation area snapshots keyed by (step, area_id)
        self._area_snapshots_by_step_area: Dict[tuple[int, int], Dict[str, Any]] = {}
        # Logger-local RNG for unbiased tie-breaks in logged top-k ranks.
        # Must be isolated from simulation RNG streams.
        seed = (int(run_seed) * 1_000_003 + int(rule_idx) * 9_173 + 17) % (2**63 - 1)
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
            cfg_dump = _safe_config_dump(config)
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
                "rule_name": getattr(model, "voting_rule_name", None) if model is not None else None,
                "rule_impl_name": getattr(model, "voting_rule_implementation_name", None) if model is not None else None,
                "distance_idx": getattr(model, "distance_idx", None) if model is not None else None,
                "distance_name": getattr(model, "distance_func_name", None) if model is not None else None,
                "distance_impl_name": getattr(model, "distance_func_implementation_name", None) if model is not None else None,
            },
            "config_ref": str(config_ref) if config_ref is not None else None,
            "config_hash": config_hash,
        }
        with open(self.ctx.out_dir / "meta.yaml", "w") as f:
            yaml.safe_dump(meta, f)

    def write_static(self, model: Model) -> None:
        """Write static.json (schema v2 metadata) and static overlay artifacts."""

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
                "names": list(getattr(model, "voting_rule_names", []) or []),
                "impl_names": list(getattr(model, "voting_rule_implementation_names", []) or []),
                "selected_idx": int(self.ctx.rule_idx),
                "selected_name": getattr(model, "voting_rule_name", None),
                "selected_impl_name": getattr(model, "voting_rule_implementation_name", None),
            },
            "distance_functions": {
                "names": list(getattr(model, "distance_func_names", []) or []),
                "impl_names": list(getattr(model, "distance_func_implementation_names", []) or []),
                "selected_idx": int(getattr(model, "distance_idx", 0)),
                "selected_name": getattr(model, "distance_func_name", None),
                "selected_impl_name": getattr(model, "distance_func_implementation_name", None),
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
        # v2 replay expects this key.
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

        - vote sink: used by `Area._tally_votes()` to emit participant vote rows
        - area snapshot sink: used by `Area._capture_area_snapshot_for_logger()`
        """
        if self._num_colors is None:
            self._num_colors = int(model.num_colors)
        model.register_schema_v2_sinks(
            vote_sink=self._on_vote,
            area_snapshot_sink=self._on_area_snapshot,
        )

    def detach_from_model(self, model: Model) -> None:
        """Detach schema-v2 sinks from the model."""
        model.clear_schema_v2_sinks()

    def log_step(self, step: int, model: Model, grid_snapshot: Optional[np.ndarray] = None) -> None:
        """Append schema-v2 rows for this step.

        Args:
            step: Recorded step number (schema v2 is 1-based).
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
        snap = getattr(model, "step_metrics_snapshot", None)
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
            "collective_assets": np.float32(float(snap["collective_assets"])),
            "gini_index": np.int16(int(snap["gini_index"])),
            "turnout": np.float32(float(snap["turnout"])),
            "mean_altruism": np.float32(float(snap["mean_altruism"])),
            "mean_dissatisfaction": np.float32(float(snap["mean_dissatisfaction"])),
        }

        pre_colors = self._get_pre_mutation_global_colors(step=step, model=model)

        if pre_colors is not None:
            for i, v in enumerate(pre_colors):
                row[f"color_{i}"] = np.float32(v)

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

        # Normalize defensively to keep a valid distribution even under tiny drift.
        return (vals / total).astype(np.float32)

    def _extract_area_steps_rows(self, step: int, model: Model) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        areas = model.areas
        num_colors = int(model.num_colors)

        options = np.asarray(model.options)

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
            area_id = int(area.unique_id)

            # Base row
            r: Dict[str, Any] = {
                "run_seed": np.int32(self.ctx.run_seed),
                "rule_idx": np.int16(self.ctx.rule_idx),
                "step": np.int32(step),
                "area_id": np.int32(area_id),
                "eligible_voters": np.int32(area.num_eligible_voters_last),
                # participants is overwritten from the pre-mutation area snapshot below.
                "participants": np.int32(0),
                "turnout": np.float32(float(area.voter_turnout)),  # In percent
                "election_cost_rate": np.float32(float(model.election_cost_rate)),
                "fee_pool": np.float32(float(area.election_fee_pool)),
                "winning_option_id": np.int32(-1),
                "dist_to_reality": np.float32(
                    float(area.dist_to_reality) if area.dist_to_reality is not None else 0.0
                ),
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
                "election_cost_rate",
                "fee_pool",
                "dist_to_reality",
                "gini_index",
                "area_color",
                "elected_color",
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
            r["election_cost_rate"] = np.float32(float(snapshot["election_cost_rate"]))
            r["fee_pool"] = np.float32(float(snapshot["fee_pool"]))
            r["dist_to_reality"] = np.float32(float(snapshot["dist_to_reality"]))
            r["gini_index"] = np.int16(int(snapshot["gini_index"]))
            voted_ordering = snapshot.get("elected_color", None)
            cd = snapshot.get("area_color", None)
            if cd is None:
                raise RuntimeError(
                    f"Missing pre-mutation area_color for step {step}, area {area_id}."
                )

            _apply_election_vectors(
                r_dict=r,
                elected_color_vec=voted_ordering,
                area_color_vec=cd,
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
                    "row": np.int16(int(a.row)),
                    "col": np.int16(int(a.col)),
                    "assets": np.float32(float(a.assets)),
                    "num_elections_participated": np.int32(int(a.num_elections_participated)),
                    "personality_group_idx": np.int16(a.personality_group_idx),
                    "eligible_for_election": bool(a.eligible_for_election),
                    "participating": bool(a.participating),
                    "election_fee": np.float32(float(a.election_fee)),
                    "reward_common_component": np.float32(float(a.reward_common_component)),
                    "reward_personal_component": np.float32(float(a.reward_personal_component)),
                    "election_delta_abs": np.float32(float(a.election_delta_abs)),
                    "election_delta_rel": np.float32(float(a.election_delta_rel)),
                    "participation_baseline": np.float32(float(a.participation_baseline)),
                    "participation_signal": np.float32(float(a.participation_signal)),
                    "altruism_factor": np.float32(float(a.altruism_factor)),
                    "dissatisfaction_value": np.float32(float(a.dissatisfaction_value)),
                    "dissatisfaction_baseline": np.float32(float(a.dissatisfaction_baseline)),
                    "dissatisfaction_signal": np.float32(float(a.dissatisfaction_signal)),
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
            return config.model_dump(mode="json")
        except (TypeError, ValueError):
            return {}
    # Pydantic v1
    if hasattr(config, "dict"):
        try:
            import json
            return json.loads(config.json())
        except (TypeError, ValueError):
            return {}
    return {}


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
