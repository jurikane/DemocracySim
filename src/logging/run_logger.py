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

This module intentionally keeps core model logic unchanged. For steps.parquet and
area_steps.parquet, color distributions are captured from pre-mutation (post-election)
snapshots. Grid snapshots are pre-mutation.
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
        # Pre-mutation area snapshots keyed by (step, area_id)
        self._area_snapshots_by_step_area: Dict[tuple[int, int], Dict[str, Any]] = {}

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

                - vote sink: used by Area._tally_votes() to emit participant vote rows.
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
            # Create an empty frame with required columns so schema validation passes
            votes_df = pd.DataFrame(
                columns=[
                    "run_seed",
                    "rule_idx",
                    "step",
                    "area_id",
                    "agent_id",
                    "participating",
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
            "collective_assets": np.float32(0.0),
            "gini_index": np.int16(0),
            "turnout": np.float32(0.0),
            "mean_altruism": np.float32(0.0),
            "mean_satisfaction": np.float32(0.0),
        }

        pre_colors = self._get_pre_mutation_global_colors(step=step, model=model)

        df = model.datacollector.get_model_vars_dataframe()
        if df is None or len(df) == 0:
            if pre_colors is not None:
                for i, v in enumerate(pre_colors):
                    row[f"color_{i}"] = np.float32(v)
            return row

        last = df.iloc[-1].to_dict()

        # Prefer snake_case (live + replay v2 use this)
        if "collective_assets" in last:
            row["collective_assets"] = np.float32(last["collective_assets"])
        elif "Collective assets" in last:
            row["collective_assets"] = np.float32(last["Collective assets"])

        if "gini_index" in last:
            row["gini_index"] = np.int16(last["gini_index"])
        elif "Gini Index (0-100)" in last:
            row["gini_index"] = np.int16(last["Gini Index (0-100)"])

        if "turnout" in last:
            row["turnout"] = np.float32(last["turnout"])
        elif "Voter turnout globally (in percent)" in last:
            row["turnout"] = np.float32(last["Voter turnout globally (in percent)"])
        if "mean_altruism" in last:
            row["mean_altruism"] = np.float32(last["mean_altruism"])
        if "mean_satisfaction" in last:
            row["mean_satisfaction"] = np.float32(last["mean_satisfaction"])

        # Optional per-color series: snake_case color_0...color_{C-1} (preferred)
        for k, v in last.items():
            if isinstance(k, str) and k.startswith("color_"):
                suf = k.split("_", 1)[1]
                if suf.isdigit():
                    row[k] = np.float32(v)

        if pre_colors is not None:
            for i, v in enumerate(pre_colors):
                row[f"color_{i}"] = np.float32(v)

        return row

    def _get_pre_mutation_global_colors(self, *, step: int, model: Model) -> Optional[np.ndarray]:
        areas = [a for a in model.areas if a is not None]
        if not areas:
            return None
        num_colors = int(model.num_colors)
        if num_colors <= 0:
            return None

        sums = np.zeros(num_colors, dtype=np.float32)
        missing: list[int] = []
        for area in areas:
            area_id = int(area.unique_id)
            snapshot = self._area_snapshots_by_step_area.get((int(step), area_id))
            if snapshot is None:
                missing.append(area_id)
                continue
            area_color = snapshot.get("area_color", None)
            if area_color is None:
                missing.append(area_id)
                continue
            cdv = np.asarray(area_color, dtype=np.float32)
            if cdv.size != num_colors:
                raise RuntimeError(
                    f"Pre-mutation snapshot for step {step} area {area_id} has "
                    f"{cdv.size} colors, expected {num_colors}."
                )
            sums += cdv

        if missing:
            raise RuntimeError(
                f"Missing pre-mutation area_color snapshot for step {step} "
                f"(areas={sorted(missing)})."
            )

        return sums / float(len(areas))

    def _extract_area_steps_rows(self, step: int, model: Model) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        areas = list(model.areas)
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
            if area is None:
                continue
            area_id = int(area.unique_id)

            # Base row
            r: Dict[str, Any] = {
                "run_seed": np.int32(self.ctx.run_seed),
                "rule_idx": np.int16(self.ctx.rule_idx),
                "step": np.int32(step),
                "area_id": np.int32(area_id),
                "eligible_voters": np.int32(area.num_agents),
                # Not tracked explicitly yet; default 0.
                "participants": np.int32(0),
                "turnout": np.float32(float(area.voter_turnout)),  # In percent
                "election_cost_rate": np.float32(float(model.election_cost_rate)),
                "fee_pool": np.float32(getattr(area, "_election_fee_pool")),
                "winning_option_id": np.int32(-1),
                "dist_to_reality": np.float32(float(area.dist_to_reality)),
                "gini_index": np.int16(0),
            }

            snapshot = self._area_snapshots_by_step_area.pop((int(step), area_id), None)
            if snapshot is None:
                raise RuntimeError(
                    f"Missing pre-mutation area snapshot for step {step}, area {area_id}."
                )
            r["eligible_voters"] = np.int32(int(snapshot.get("eligible_voters", area.num_agents)))
            r["participants"] = np.int32(int(snapshot.get("participants", 0)))
            r["turnout"] = np.float32(float(snapshot.get("turnout", area.voter_turnout)))
            r["election_cost_rate"] = np.float32(float(snapshot.get("election_cost_rate", model.election_cost_rate)))
            r["fee_pool"] = np.float32(float(snapshot.get("fee_pool", getattr(area, "_election_fee_pool"))))
            r["dist_to_reality"] = np.float32(float(snapshot.get("dist_to_reality", area.dist_to_reality)))
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

            # area gini from agents' assets (same as old replay logger)
            agents = list(area.agents)
            if agents:
                assets = [float(a.assets) for a in agents]
                # reuse metric helper indirectly: gini_index_0_100 exists in utils.metrics
                from src.utils.metrics import gini_index_0_100

                r["gini_index"] = np.int16(int(gini_index_0_100(assets)))

            rows.append(r)

        return rows

    def _extract_agent_rows(self, step: int, model: Model) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        agents = list(model.voting_agents)
        for a in agents:
            if a is None:
                continue
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
                    "participation_baseline": np.float32(float(a.participation_baseline)),
                    "participation_signal": np.float32(float(a.participation_signal)),
                    "altruism_factor": np.float32(float(a.altruism_factor)),
                    "satisfaction_value": np.float32(float(a.satisfaction_value)),
                    "satisfaction_baseline": np.float32(float(a.satisfaction_baseline)),
                    "satisfaction_signal": np.float32(float(a.satisfaction_signal)),
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
        # For ties, use a stable secondary sort by option id for determinism.
        order = np.lexsort((agent.model.option_vec, scores))
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
        if dist is None or dist.ndim != 1:
            # Emit NaNs to avoid masking missing estimate_real_distribution()
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
