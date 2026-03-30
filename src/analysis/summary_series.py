from __future__ import annotations

from math import factorial, log
from typing import Any
import itertools

import numpy as np
import pandas as pd

from src.analysis.quality_distance import quality_distance_source, resolve_quality_distance_series
from src.analysis.reference_benchmarks import l1_dist
from src.utils.ballots import score_options_c2
from src.utils.distance_functions import kendall_tau_order, spearman_fr_order
from src.utils.metrics import gini_index_0_100
from src.utils.representations import distribution_to_ordering_tie_aware
from src.utils.social_welfare_functions import approval_voting, borda_rule, plurality_rule, random_rule, schulze_rule, utilitarian_rule

_MODE_ALIGNMENT_LOW_SUPPORT_VOTES = 5
_SMALL_GROUP_MIN_RESIDENTS = 5

def _build_global_series(
    *,
    steps: pd.DataFrame,
    area_steps: pd.DataFrame,
    agents: pd.DataFrame,
    votes: pd.DataFrame,
    num_colors: int,
    refs_global: dict[str, np.ndarray | None],
    quality_target_mode: str,
) -> pd.DataFrame:
    color_cols = [f"color_{i}" for i in range(num_colors)]
    missing_color_cols = [c for c in color_cols if c not in steps.columns]
    if missing_color_cols:
        raise RuntimeError(f"steps.parquet missing required color columns: {missing_color_cols}")

    g = pd.DataFrame(
        {
            "step": steps["step"].astype(np.int32),
            "turnout": steps["turnout"].astype(np.float32),
            "gini_assets": steps["gini_index"].astype(np.float32),
            "mean_dissatisfaction": steps["mean_dissatisfaction"].astype(np.float32),
        }
    )
    if "winning_option_id" in steps.columns:
        g["winning_option_id"] = steps["winning_option_id"].astype(np.int32)
    elif "winning_option_id" in area_steps.columns and "area_id" in area_steps.columns:
        # Global winner ordering is only well-defined in single-area runs.
        unique_areas = area_steps["area_id"].dropna().astype(int).unique().tolist()
        if len(unique_areas) == 1:
            step_to_win = (
                area_steps[["step", "winning_option_id"]]
                .drop_duplicates(subset=["step"], keep="first")
                .set_index("step")["winning_option_id"]
            )
            g["winning_option_id"] = (
                g["step"].map(step_to_win).fillna(-1).astype(np.int32)
            )
    for c in color_cols:
        g[c] = steps[c].astype(np.float32)

    gini_diss = (
        agents.groupby("step", sort=True)["dissatisfaction_value"]
        .apply(lambda s: float(gini_index_0_100(s.to_numpy(dtype=float))))
        .reindex(g["step"].to_numpy(dtype=int), fill_value=np.nan)
        .to_numpy(dtype=float)
    )
    g["gini_dissatisfaction"] = np.asarray(gini_diss, dtype=np.float32)

    g["dist_to_reality"] = np.asarray(
        _weighted_dist_to_reality_by_step(area_steps=area_steps, step_index=g["step"].to_numpy(dtype=int)),
        dtype=np.float32,
    )
    area_steps_q = area_steps.copy()
    area_steps_q["quality_distance"] = resolve_quality_distance_series(
        area_steps_q,
        quality_target_mode=quality_target_mode,
        run_label="summary_tooling:_build_global_series",
    )
    g["quality_distance"] = np.asarray(
        _weighted_metric_by_step(
            area_steps=area_steps_q,
            step_index=g["step"].to_numpy(dtype=int),
            value_col="quality_distance",
        ),
        dtype=np.float32,
    )
    puzzle_color_cols = [f"puzzle_color_{i}" for i in range(num_colors) if f"puzzle_color_{i}" in area_steps.columns]
    if len(puzzle_color_cols) == num_colors:
        weighted_puzzle = _weighted_vector_metric_by_step(
            area_steps=area_steps,
            step_index=g["step"].to_numpy(dtype=int),
            value_cols=puzzle_color_cols,
        )
        for i, c in enumerate(puzzle_color_cols):
            g[c] = weighted_puzzle[:, i].astype(np.float32)
    if "puzzle_distance" in area_steps.columns:
        g["puzzle_distance"] = np.asarray(
            _weighted_metric_by_step(
                area_steps=area_steps,
                step_index=g["step"].to_numpy(dtype=int),
                value_col="puzzle_distance",
            ),
            dtype=np.float32,
        )
    else:
        g["puzzle_distance"] = np.float32(np.nan)

    g["diversity_first_choice_entropy"] = np.asarray(
        _diversity_entropy_by_step(
            votes=votes,
            step_index=g["step"].to_numpy(dtype=int),
            num_options=factorial(num_colors),
        ),
        dtype=np.float32,
    )

    # Benchmark-layer references (post-run analysis):
    # utilitarian=L2^2 mean, nash=KL geometric mean, rawlsian=minimax L2^2,
    # egalitarian=mean(z)+lambda*Gini(z) with lambda in {0.25,1,4}.
    colors = g[color_cols].to_numpy(dtype=np.float64)
    for key in (
        "dist_to_ref_utilitarian",
        "dist_to_ref_nash",
        "dist_to_ref_rawlsian",
        "dist_to_ref_egalitarian",
        "dist_to_ref_egalitarian_lam025",
        "dist_to_ref_egalitarian_lam400",
    ):
        ref = refs_global.get(key)
        if ref is None:
            g[key] = np.float32(np.nan)
        else:
            g[key] = np.asarray([l1_dist(row, ref) for row in colors], dtype=np.float32)
    return g

def _build_area_series(
    *,
    area_steps: pd.DataFrame,
    votes: pd.DataFrame,
    agents: pd.DataFrame,
    area_agent_ids: dict[int, list[int]],
    num_colors: int,
    refs_by_area: dict[int, dict[str, np.ndarray | None]],
    quality_target_mode: str,
) -> pd.DataFrame:
    area_color_cols = [f"area_color_{i}" for i in range(num_colors)]
    missing = [c for c in area_color_cols if c not in area_steps.columns]
    if missing:
        raise RuntimeError(f"area_steps.parquet missing required area_color columns: {missing}")

    a = pd.DataFrame(
        {
            "step": area_steps["step"].astype(np.int32),
            "area_id": area_steps["area_id"].astype(np.int32),
            "winning_option_id": area_steps["winning_option_id"].astype(np.int32),
            "participants": area_steps["participants"].astype(np.int32),
            "eligible_voters": area_steps["eligible_voters"].astype(np.int32),
            "turnout": area_steps["turnout"].astype(np.float32),
            "gini_assets": area_steps["gini_index"].astype(np.float32),
            "dist_to_reality": area_steps["dist_to_reality"].astype(np.float32),
        }
    )
    if "puzzle_distance" in area_steps.columns:
        a["puzzle_distance"] = area_steps["puzzle_distance"].astype(np.float32)
    else:
        a["puzzle_distance"] = np.float32(np.nan)
    a["quality_distance"] = np.asarray(
        resolve_quality_distance_series(
            area_steps,
            quality_target_mode=quality_target_mode,
            run_label="summary_tooling:_build_area_series",
        ),
        dtype=np.float32,
    )
    if "grid_ordering_id" in area_steps.columns:
        a["grid_ordering_id"] = area_steps["grid_ordering_id"].astype(np.int32)
    if "puzzle_ordering_id" in area_steps.columns:
        a["puzzle_ordering_id"] = area_steps["puzzle_ordering_id"].astype(np.int32)
    for c in area_color_cols:
        a[c] = area_steps[c].astype(np.float32)
    for c in [f"puzzle_color_{i}" for i in range(num_colors) if f"puzzle_color_{i}" in area_steps.columns]:
        a[c] = area_steps[c].astype(np.float32)

    area_gini_diss = _compute_area_gini_dissatisfaction(
        agents=agents,
        area_agent_ids=area_agent_ids,
    )
    if not area_gini_diss.empty:
        a = a.merge(area_gini_diss, on=["step", "area_id"], how="left")
    else:
        a["gini_dissatisfaction"] = np.float32(np.nan)

    diversity_rows = _diversity_entropy_by_step_area(votes=votes, num_options=factorial(num_colors))
    if diversity_rows.empty:
        a["diversity_first_choice_entropy"] = np.float32(np.nan)
    else:
        a = a.merge(diversity_rows, on=["step", "area_id"], how="left")
        a["diversity_first_choice_entropy"] = a["diversity_first_choice_entropy"].astype(np.float32)

    mode_alignment = _compute_area_vote_mode_alignment_series(
        area_steps=area_steps,
        votes=votes,
        num_colors=num_colors,
    )
    if not mode_alignment.empty:
        a = a.merge(mode_alignment, on=["step", "area_id"], how="left")
    else:
        for c in (
            "altruistic_rank1_match_puzzle_share",
            "self_regarding_rank1_match_puzzle_share",
            "altruistic_rank1_match_outcome_share",
            "self_regarding_rank1_match_outcome_share",
            "altruistic_votes_count",
            "self_regarding_votes_count",
            "vote_count_total",
            "altruistic_vote_share",
            "self_regarding_vote_share",
        ):
            a[c] = np.float32(np.nan)

    area_color_cols = [f"area_color_{i}" for i in range(num_colors)]
    for area_id in sorted(a["area_id"].unique().tolist()):
        mask = a["area_id"].to_numpy(dtype=int) == int(area_id)
        refs = refs_by_area.get(int(area_id), {})
        colors = a.loc[mask, area_color_cols].to_numpy(dtype=np.float64)
        for key in (
            "dist_to_ref_utilitarian",
            "dist_to_ref_nash",
            "dist_to_ref_rawlsian",
            "dist_to_ref_egalitarian",
            "dist_to_ref_egalitarian_lam025",
            "dist_to_ref_egalitarian_lam400",
        ):
            ref = refs.get(key)
            if ref is None:
                a.loc[mask, key] = np.float32(np.nan)
            else:
                a.loc[mask, key] = np.asarray([l1_dist(row, ref) for row in colors], dtype=np.float32)

    return a.sort_values(["step", "area_id"]).reset_index(drop=True)

def _compute_area_vote_mode_alignment_series(
    *,
    area_steps: pd.DataFrame,
    votes: pd.DataFrame,
    num_colors: int,
) -> pd.DataFrame:
    """Per-area, per-step vote-mode alignment shares (rank-1 matches) to puzzle/outcome."""
    if votes.empty or num_colors <= 0:
        return pd.DataFrame(
            columns=[
                "step",
                "area_id",
                "altruistic_rank1_match_puzzle_share",
                "self_regarding_rank1_match_puzzle_share",
                "altruistic_rank1_match_outcome_share",
                "self_regarding_rank1_match_outcome_share",
                "altruistic_votes_count",
                "self_regarding_votes_count",
                "vote_count_total",
                "altruistic_vote_share",
                "self_regarding_vote_share",
            ]
        )
    req_vote = {"step", "area_id", "agent_id", "rank_1_option_id", "voted_altruistically"}
    req_area = {"step", "area_id", "winning_option_id"}
    if not req_vote.issubset(votes.columns) or not req_area.issubset(area_steps.columns):
        return pd.DataFrame()

    v = (
        votes[["step", "area_id", "agent_id", "rank_1_option_id", "voted_altruistically"]]
        .drop_duplicates(subset=["step", "area_id", "agent_id"], keep="first")
        .copy()
    )
    v = v[v["voted_altruistically"].isin([True, False])].copy()
    if v.empty:
        return pd.DataFrame()
    v["step"] = v["step"].astype("int32")
    v["area_id"] = v["area_id"].astype("int32")
    v["rank_1_option_id"] = v["rank_1_option_id"].astype("Int32")
    v = v.dropna(subset=["rank_1_option_id"])
    if v.empty:
        return pd.DataFrame()
    v["rank_1_option_id"] = v["rank_1_option_id"].astype("int32")

    base = area_steps[["step", "area_id", "winning_option_id"]].copy()
    base["step"] = base["step"].astype("int32")
    base["area_id"] = base["area_id"].astype("int32")
    base["winning_option_id"] = base["winning_option_id"].astype("int32")

    # Puzzle option id requires puzzle distribution vectors; older runs may not have them.
    puzzle_cols = [f"puzzle_color_{i}" for i in range(num_colors) if f"puzzle_color_{i}" in area_steps.columns]
    base["puzzle_option_id"] = np.int32(-1)
    if "puzzle_ordering_id" in area_steps.columns:
        base["puzzle_option_id"] = area_steps["puzzle_ordering_id"].fillna(-1).astype("int32")
    elif len(puzzle_cols) == num_colors:
        options = np.asarray(list(itertools.permutations(range(int(num_colors)))), dtype=np.int64)
        option_lookup = {tuple(int(x) for x in row.tolist()): int(i) for i, row in enumerate(options)}
        rows = []
        tie_rng = np.random.default_rng(20240229)
        for area_id, block in area_steps.sort_values(["area_id", "step"]).groupby("area_id", sort=True):
            prev_ord = None
            for _, r in block.iterrows():
                vals = np.asarray([float(r[c]) for c in puzzle_cols], dtype=np.float64)
                if np.isfinite(vals).all():
                    ord_p = _summary_ordering_from_distribution_tie_aware(vals, reference_ordering=prev_ord, rng=tie_rng)
                    prev_ord = ord_p
                    oid = option_lookup.get(tuple(int(x) for x in ord_p.tolist()), -1)
                else:
                    oid = -1
                rows.append((int(r["step"]), int(area_id), int(oid)))
        if rows:
            p_df = pd.DataFrame(rows, columns=["step", "area_id", "puzzle_option_id"]).astype(
                {"step": "int32", "area_id": "int32", "puzzle_option_id": "int32"}
            )
            base = base.drop(columns=["puzzle_option_id"]).merge(p_df, on=["step", "area_id"], how="left")
            base["puzzle_option_id"] = base["puzzle_option_id"].fillna(-1).astype("int32")

    v = v.merge(base, on=["step", "area_id"], how="left")
    if v.empty:
        return pd.DataFrame()
    v["match_outcome"] = (v["rank_1_option_id"].to_numpy(dtype=int) == v["winning_option_id"].to_numpy(dtype=int)).astype("int8")
    v["match_puzzle"] = (
        (v["puzzle_option_id"].to_numpy(dtype=int) >= 0)
        & (v["rank_1_option_id"].to_numpy(dtype=int) == v["puzzle_option_id"].to_numpy(dtype=int))
    ).astype("int8")

    rows_out: list[dict[str, Any]] = []
    for (step, area_id), block in v.groupby(["step", "area_id"], sort=True):
        row: dict[str, Any] = {"step": int(step), "area_id": int(area_id)}
        total_votes = int(len(block))
        row["vote_count_total"] = int(total_votes)
        for mode_val, prefix in ((True, "altruistic"), (False, "self_regarding")):
            m = block[block["voted_altruistically"] == mode_val]
            mode_count = int(len(m))
            row[f"{prefix}_votes_count"] = int(mode_count)
            row[f"{prefix}_vote_share"] = (
                np.float32(100.0 * (float(mode_count) / float(total_votes)))
                if total_votes > 0
                else np.float32(np.nan)
            )
            if len(m) > 0:
                row[f"{prefix}_rank1_match_outcome_share"] = np.float32(100.0 * float(m["match_outcome"].mean()))
                # Only meaningful if puzzle option id exists for that row's step.
                valid_puzzle = m["puzzle_option_id"].to_numpy(dtype=int) >= 0
                if bool(np.any(valid_puzzle)):
                    row[f"{prefix}_rank1_match_puzzle_share"] = np.float32(
                        100.0 * float(m.loc[valid_puzzle, "match_puzzle"].mean())
                    )
                else:
                    row[f"{prefix}_rank1_match_puzzle_share"] = np.float32(np.nan)
            else:
                row[f"{prefix}_rank1_match_outcome_share"] = np.float32(np.nan)
                row[f"{prefix}_rank1_match_puzzle_share"] = np.float32(np.nan)
        rows_out.append(row)
    return pd.DataFrame(rows_out)

def _compute_area_gini_dissatisfaction(*, agents: pd.DataFrame, area_agent_ids: dict[int, list[int]]) -> pd.DataFrame:
    if not area_agent_ids:
        return pd.DataFrame(columns=["step", "area_id", "gini_dissatisfaction"])
    needed = {"step", "agent_id", "dissatisfaction_value"}
    if not needed.issubset(agents.columns):
        return pd.DataFrame(columns=["step", "area_id", "gini_dissatisfaction"])

    base = agents[["step", "agent_id", "dissatisfaction_value"]].copy()
    base["step"] = base["step"].astype("int32")
    base["agent_id"] = base["agent_id"].astype("int32")
    base["dissatisfaction_value"] = base["dissatisfaction_value"].astype("float32")
    by_step: dict[int, pd.DataFrame] = {int(s): b for s, b in base.groupby("step", sort=True)}

    rows: list[dict[str, float]] = []
    for step, block in by_step.items():
        vals_by_agent = dict(zip(block["agent_id"].tolist(), block["dissatisfaction_value"].tolist()))
        for area_id, ids in area_agent_ids.items():
            vals = [float(vals_by_agent[aid]) for aid in ids if aid in vals_by_agent]
            if not vals:
                g = float("nan")
            else:
                g = float(gini_index_0_100(vals))
            rows.append({"step": int(step), "area_id": int(area_id), "gini_dissatisfaction": np.float32(g)})
    return pd.DataFrame(rows)

def _build_area_group_series(
    *,
    agents: pd.DataFrame,
    votes: pd.DataFrame,
    area_agent_ids: dict[int, list[int]],
    participation_alpha: float = 1.0,
    participation_signal_mode: str = "raw_delta_rel",
    participation_signal_group_shrink_k: float = 0.0,
    altruism_alpha: float = 1.0,
    altruism_learning: bool = True,
) -> pd.DataFrame:
    cols = [
        "step",
        "area_id",
        "group_idx",
        "residents",
        "eligible",
        "participants",
        "self_regarding_voters",
        "turnout",
        "mean_assets",
        "mean_dissatisfaction",
        "resident_share",
        "eligible_share",
        "participant_share",
        "group_reward_count",
        "group_punishment_count",
        "participants_reward_count",
        "participants_punishment_count",
        "abstainers_reward_count",
        "abstainers_punishment_count",
        "learning_direction_score",
        "learning_intensity_rel_mean",
        "learning_signed_pressure",
        "participants_mean_delta_rel",
        "abstainers_mean_delta_rel",
        "participants_mean_fee",
        "participants_mean_fee_over_assets",
        "group_mu_delta_rel",
        "global_mu_delta_rel",
        "group_signal_shrink_weight",
        "group_signal_component",
        "participants_mean_participation_signal",
        "abstainers_mean_participation_signal",
        "participants_mean_signal_group_component",
        "participants_mean_signal_fee_component",
        "abstainers_mean_signal_group_component",
        "abstainers_mean_signal_fee_component",
        "group_mean_participation_q_delta",
        "participants_mean_participation_q_delta",
        "abstainers_mean_participation_q_delta",
        "altruistic_voters_mean_participation_q_delta",
        "self_regarding_voters_mean_participation_q_delta",
        "group_mean_participation_p_delta",
        "group_std_q_participation",
        "group_std_participation_probability",
        "group_gini_assets_within",
        "group_gini_dissatisfaction_within",
        "participants_mean_participation_p_delta",
        "abstainers_mean_participation_p_delta",
        "group_mean_participation_q_update_proxy",
        "participants_mean_participation_q_update_proxy",
        "abstainers_mean_participation_q_update_proxy",
        "altruistic_voters_mean_participation_q_update_proxy",
        "self_regarding_voters_mean_participation_q_update_proxy",
        "altruistic_voters_mean_dissatisfaction_signal",
        "self_regarding_voters_mean_dissatisfaction_signal",
        "altruistic_voters_mean_altruism_delta",
        "self_regarding_voters_mean_altruism_delta",
        "altruistic_voters_mean_altruism_update_proxy",
        "self_regarding_voters_mean_altruism_update_proxy",
        "vote_mode_switch_share",
        "vote_mode_switch_from_altruistic_share",
        "vote_mode_switch_from_self_regarding_share",
        "participation_switch_to_abstain_share",
    ]
    if not area_agent_ids:
        return pd.DataFrame(columns=cols)

    needed_agents = {"step", "agent_id", "personality_group_idx", "assets", "dissatisfaction_value"}
    if not needed_agents.issubset(agents.columns):
        return pd.DataFrame(columns=cols)

    # Static agent->group map (group is immutable by model design).
    ag = (
        agents[["agent_id", "personality_group_idx"]]
        .drop_duplicates(subset=["agent_id"], keep="first")
        .astype({"agent_id": "int32", "personality_group_idx": "int16"})
    )

    resident_rows: list[dict[str, int]] = []
    for area_id, ids in sorted(area_agent_ids.items()):
        for agent_id in ids:
            resident_rows.append({"area_id": int(area_id), "agent_id": int(agent_id)})
    if not resident_rows:
        return pd.DataFrame(columns=cols)

    residents = pd.DataFrame(resident_rows).astype({"area_id": "int32", "agent_id": "int32"})
    residents = residents.merge(ag, on="agent_id", how="inner")
    if residents.empty:
        return pd.DataFrame(columns=cols)

    step_df = pd.DataFrame({"step": sorted(int(v) for v in agents["step"].dropna().unique().tolist())}).astype({"step": "int32"})
    residents = residents.assign(_k=1).merge(step_df.assign(_k=1), on="_k", how="inner").drop(columns="_k")

    state_cols = ["step", "agent_id", "assets", "dissatisfaction_value"]
    optional_cols = [
        "participating",
        "election_fee",
        "election_delta_abs",
        "election_delta_rel",
        "participation_signal",
        "participation_signal_group_component",
        "participation_signal_fee_component",
        "dissatisfaction_signal",
        "q_participation",
        "participation_probability",
        "altruism_factor",
    ]
    for c in optional_cols:
        if c in agents.columns:
            state_cols.append(c)
    state = agents[state_cols].copy()
    state["step"] = state["step"].astype("int32")
    state["agent_id"] = state["agent_id"].astype("int32")
    state["assets"] = state["assets"].astype("float32")
    state["dissatisfaction_value"] = state["dissatisfaction_value"].astype("float32")
    if "participating" not in state.columns:
        state["participating"] = False
    if "election_fee" not in state.columns:
        state["election_fee"] = np.nan
    if "election_delta_abs" not in state.columns:
        state["election_delta_abs"] = np.nan
    if "election_delta_rel" not in state.columns:
        state["election_delta_rel"] = np.nan
    if "participation_signal" not in state.columns:
        state["participation_signal"] = np.nan
    if "participation_signal_group_component" not in state.columns:
        state["participation_signal_group_component"] = np.nan
    if "participation_signal_fee_component" not in state.columns:
        state["participation_signal_fee_component"] = np.nan
    if "dissatisfaction_signal" not in state.columns:
        state["dissatisfaction_signal"] = np.nan
    if "q_participation" not in state.columns:
        state["q_participation"] = np.nan
    if "participation_probability" not in state.columns:
        state["participation_probability"] = np.nan
    if "altruism_factor" not in state.columns:
        state["altruism_factor"] = np.nan
    state["participating"] = state["participating"].astype("boolean").fillna(False).astype(bool)
    state["election_fee"] = state["election_fee"].astype("float32")
    state["election_delta_abs"] = state["election_delta_abs"].astype("float32")
    state["election_delta_rel"] = state["election_delta_rel"].astype("float32")
    state["participation_signal"] = state["participation_signal"].astype("float32")
    state["participation_signal_group_component"] = state["participation_signal_group_component"].astype("float32")
    state["participation_signal_fee_component"] = state["participation_signal_fee_component"].astype("float32")
    state["dissatisfaction_signal"] = state["dissatisfaction_signal"].astype("float32")
    state["q_participation"] = state["q_participation"].astype("float32")
    state["participation_probability"] = state["participation_probability"].astype("float32")
    state["altruism_factor"] = state["altruism_factor"].astype("float32")
    residents = residents.merge(state, on=["step", "agent_id"], how="left")

    grouped = (
        residents.groupby(["step", "area_id", "personality_group_idx"], sort=True, as_index=False)
        .agg(
            residents=("agent_id", "size"),
            eligible=("assets", lambda s: int(np.sum(np.asarray(s, dtype=float) > 0.0))),
            mean_assets=("assets", "mean"),
            mean_dissatisfaction=("dissatisfaction_value", "mean"),
            group_std_q_participation=("q_participation", "std"),
            group_std_participation_probability=("participation_probability", "std"),
        )
    )

    gini_rows: list[dict[str, Any]] = []
    for (step, area_id, gi), block in residents.groupby(
        ["step", "area_id", "personality_group_idx"], sort=False
    ):
        a_vals = pd.to_numeric(block["assets"], errors="coerce").dropna().to_numpy(dtype=float)
        d_vals = pd.to_numeric(block["dissatisfaction_value"], errors="coerce").dropna().to_numpy(dtype=float)
        gini_rows.append(
            {
                "step": int(step),
                "area_id": int(area_id),
                "personality_group_idx": int(gi),
                "group_gini_assets_within": np.float32(gini_index_0_100(a_vals) if a_vals.size > 0 else np.nan),
                "group_gini_dissatisfaction_within": np.float32(
                    gini_index_0_100(d_vals) if d_vals.size > 0 else np.nan
                ),
            }
        )
    if gini_rows:
        grouped = grouped.merge(
            pd.DataFrame(gini_rows),
            on=["step", "area_id", "personality_group_idx"],
            how="left",
        )

    if votes.empty:
        participants = pd.DataFrame(
            columns=[
                "step",
                "area_id",
                "personality_group_idx",
                "participants",
                "self_regarding_voters",
            ]
        )
    else:
        vote_cols = ["step", "area_id", "agent_id"]
        has_vote_mode = "voted_altruistically" in votes.columns
        if has_vote_mode:
            vote_cols.append("voted_altruistically")
        v = votes[vote_cols].drop_duplicates(subset=["step", "area_id", "agent_id"], keep="first").copy()
        v["step"] = v["step"].astype("int32")
        v["area_id"] = v["area_id"].astype("int32")
        v["agent_id"] = v["agent_id"].astype("int32")
        v = v.merge(ag, on="agent_id", how="left")
        v = v.dropna(subset=["personality_group_idx"])
        if has_vote_mode:
            v["self_regarding_voters"] = (v["voted_altruistically"] == False).astype("int32")
        else:
            v["self_regarding_voters"] = 0
        participants = (
            v.groupby(["step", "area_id", "personality_group_idx"], sort=True, as_index=False)
            .agg(
                participants=("agent_id", "nunique"),
                self_regarding_voters=("self_regarding_voters", "sum"),
            )
        )

    out = grouped.merge(
        participants,
        on=["step", "area_id", "personality_group_idx"],
        how="left",
    )
    out["participants"] = out["participants"].fillna(0).astype("int32")
    out["self_regarding_voters"] = out["self_regarding_voters"].fillna(0).astype("int32")
    out["turnout"] = np.where(
        out["residents"].to_numpy(dtype=float) > 0.0,
        (out["participants"].to_numpy(dtype=float) / out["residents"].to_numpy(dtype=float)) * 100.0,
        np.nan,
    )
    out["turnout"] = out["turnout"].astype("float32")

    totals = (
        out.groupby(["step", "area_id"], sort=False, as_index=False)
        .agg(
            residents_total=("residents", "sum"),
            eligible_total=("eligible", "sum"),
            participants_total=("participants", "sum"),
        )
    )
    out = out.merge(totals, on=["step", "area_id"], how="left")
    residents_total = out["residents_total"].to_numpy(dtype=float)
    eligible_total = out["eligible_total"].to_numpy(dtype=float)
    participants_total = out["participants_total"].to_numpy(dtype=float)

    resident_share = np.full(len(out), np.nan, dtype=np.float32)
    np.divide(
        out["residents"].to_numpy(dtype=float),
        residents_total,
        out=resident_share,
        where=residents_total > 0.0,
    )
    eligible_share = np.full(len(out), np.nan, dtype=np.float32)
    np.divide(
        out["eligible"].to_numpy(dtype=float),
        eligible_total,
        out=eligible_share,
        where=eligible_total > 0.0,
    )
    participant_share = np.zeros(len(out), dtype=np.float32)
    np.divide(
        out["participants"].to_numpy(dtype=float),
        participants_total,
        out=participant_share,
        where=participants_total > 0.0,
    )

    out["resident_share"] = resident_share
    out["eligible_share"] = eligible_share
    out["participant_share"] = participant_share

    # Participation-learning causal decomposition anchors:
    # group mean delta_rel, cross-group mean, shrink weight, and implied group signal component.
    eligible_mask_for_mu = residents["assets"].to_numpy(dtype=float) > 0.0
    mu_by_group = (
        residents.loc[eligible_mask_for_mu]
        .groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)["election_delta_rel"]
        .mean()
        .rename(columns={"election_delta_rel": "group_mu_delta_rel"})
    )
    mu_global = (
        mu_by_group.groupby(["step", "area_id"], sort=False, as_index=False)["group_mu_delta_rel"]
        .mean()
        .rename(columns={"group_mu_delta_rel": "global_mu_delta_rel"})
    )
    out = out.merge(mu_by_group, on=["step", "area_id", "personality_group_idx"], how="left")
    out = out.merge(mu_global, on=["step", "area_id"], how="left")

    shrink_k = float(participation_signal_group_shrink_k) if np.isfinite(participation_signal_group_shrink_k) else np.nan
    eligible_vals = out["eligible"].to_numpy(dtype=float)
    if np.isfinite(shrink_k) and shrink_k >= 0.0:
        shrink_weight = np.divide(
            eligible_vals,
            eligible_vals + float(shrink_k),
            out=np.full(len(out), np.nan, dtype=float),
            where=(eligible_vals + float(shrink_k)) > 0.0,
        )
    else:
        shrink_weight = np.full(len(out), np.nan, dtype=float)
    out["group_signal_shrink_weight"] = shrink_weight.astype("float32")
    mu_group_vals = pd.to_numeric(out.get("group_mu_delta_rel"), errors="coerce").to_numpy(dtype=float)
    mu_global_vals = pd.to_numeric(out.get("global_mu_delta_rel"), errors="coerce").to_numpy(dtype=float)
    out["group_signal_component"] = (shrink_weight * (mu_group_vals - mu_global_vals)).astype("float32")

    # Participant/abstainer incentive diagnostics for calibration plots.
    residents = residents.sort_values(["area_id", "agent_id", "step"]).reset_index(drop=True)
    grp_agent = residents.groupby(["area_id", "agent_id"], sort=False)
    for src_col, dst_col in (
        ("q_participation", "participation_q_delta"),
        ("participation_probability", "participation_p_delta"),
        ("altruism_factor", "altruism_delta"),
    ):
        prev = grp_agent[src_col].shift(1).to_numpy(dtype=float)
        cur = residents[src_col].to_numpy(dtype=float)
        residents[dst_col] = (cur - prev).astype("float32")

    residents["fee_over_assets"] = np.where(
        residents["assets"].to_numpy(dtype=float) > 0.0,
        residents["election_fee"].to_numpy(dtype=float) / residents["assets"].to_numpy(dtype=float),
        np.nan,
    ).astype("float32")
    alpha = float(participation_alpha) if np.isfinite(participation_alpha) else 1.0
    mode = str(participation_signal_mode).strip()
    resident_eligible = residents["assets"].to_numpy(dtype=float) > 0.0
    p_signal = residents["participation_signal"].to_numpy(dtype=float)
    if mode == "group_relative_delta_rel_party":
        q_push_proxy = p_signal
    else:
        q_push_proxy = np.where(
            residents["participating"].astype(bool).to_numpy(),
            1.0,
            -1.0,
        ) * p_signal
    residents["participation_q_update_proxy"] = np.where(
        resident_eligible,
        alpha * q_push_proxy,
        np.nan,
    ).astype("float32")
    altruism_alpha_eff = float(altruism_alpha) if np.isfinite(altruism_alpha) else 1.0
    residents["altruism_update_proxy"] = np.where(
        bool(altruism_learning),
        altruism_alpha_eff * residents["dissatisfaction_signal"].to_numpy(dtype=float),
        np.nan,
    ).astype("float32")
    p_mask = residents["participating"].astype(bool).to_numpy()
    a_mask = ~p_mask
    delta_abs = residents["election_delta_abs"].to_numpy(dtype=float)
    residents["is_reward"] = (delta_abs > 0.0).astype("int8")
    residents["is_punishment"] = (delta_abs < 0.0).astype("int8")

    # Learning-aligned diagnostics:
    # direction follows action/outcome sign logic, intensity is normalized by
    # step-global mean |delta_abs| for the same action/outcome channel.
    mask_part_reward = p_mask & (delta_abs > 0.0)
    mask_part_punish = p_mask & (delta_abs < 0.0)
    mask_abs_reward = a_mask & (delta_abs > 0.0)
    mask_abs_punish = a_mask & (delta_abs < 0.0)
    learning_direction = np.zeros(len(residents), dtype=float)
    learning_direction[mask_part_reward] = 1.0
    learning_direction[mask_part_punish] = -1.0
    learning_direction[mask_abs_reward] = -1.0
    learning_direction[mask_abs_punish] = 1.0
    residents["learning_direction"] = learning_direction.astype("float32")

    def _global_abs_delta_mean(mask: np.ndarray) -> pd.Series:
        cols_local = ["step", "area_id", "election_delta_abs"]
        tmp = residents.loc[mask, cols_local].copy()
        if tmp.empty:
            return pd.Series(dtype=float)
        tmp["delta_abs_mag"] = pd.to_numeric(tmp["election_delta_abs"], errors="coerce").abs()
        return tmp.groupby(["step", "area_id"], sort=False)["delta_abs_mag"].mean()

    idx_step_area = pd.MultiIndex.from_arrays([residents["step"], residents["area_id"]])
    g_part_reward = _global_abs_delta_mean(mask_part_reward).reindex(idx_step_area).to_numpy(dtype=float)
    g_part_punish = _global_abs_delta_mean(mask_part_punish).reindex(idx_step_area).to_numpy(dtype=float)
    g_abs_reward = _global_abs_delta_mean(mask_abs_reward).reindex(idx_step_area).to_numpy(dtype=float)
    g_abs_punish = _global_abs_delta_mean(mask_abs_punish).reindex(idx_step_area).to_numpy(dtype=float)

    denom = np.full(len(residents), np.nan, dtype=float)
    denom[mask_part_reward] = g_part_reward[mask_part_reward]
    denom[mask_part_punish] = g_part_punish[mask_part_punish]
    denom[mask_abs_reward] = g_abs_reward[mask_abs_reward]
    denom[mask_abs_punish] = g_abs_punish[mask_abs_punish]

    abs_delta = np.abs(delta_abs)
    learning_intensity_rel = np.zeros(len(residents), dtype=float)
    valid_intensity = np.isfinite(abs_delta) & np.isfinite(denom) & (denom > 0.0)
    learning_intensity_rel[valid_intensity] = abs_delta[valid_intensity] / denom[valid_intensity]
    residents["learning_intensity_rel"] = learning_intensity_rel.astype("float32")
    residents["learning_signed_pressure"] = (learning_direction * learning_intensity_rel).astype("float32")

    reward_counts = (
        residents.groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)
        .agg(
            group_reward_count=("is_reward", "sum"),
            group_punishment_count=("is_punishment", "sum"),
        )
    )
    participant_reward_counts = (
        residents.loc[p_mask]
        .groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)
        .agg(
            participants_reward_count=("is_reward", "sum"),
            participants_punishment_count=("is_punishment", "sum"),
        )
    )
    abstainer_reward_counts = (
        residents.loc[a_mask]
        .groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)
        .agg(
            abstainers_reward_count=("is_reward", "sum"),
            abstainers_punishment_count=("is_punishment", "sum"),
        )
    )

    p_delta_rel = (
        residents.loc[p_mask]
        .groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)["election_delta_rel"]
        .mean()
        .rename(columns={"election_delta_rel": "participants_mean_delta_rel"})
    )
    a_delta = (
        residents.loc[a_mask]
        .groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)["election_delta_rel"]
        .mean()
        .rename(columns={"election_delta_rel": "abstainers_mean_delta_rel"})
    )
    p_fee = (
        residents.loc[p_mask]
        .groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)
        .agg(
            participants_mean_fee=("election_fee", "mean"),
            participants_mean_fee_over_assets=("fee_over_assets", "mean"),
        )
    )
    p_sig = (
        residents.loc[p_mask]
        .groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)["participation_signal"]
        .mean()
        .rename(columns={"participation_signal": "participants_mean_participation_signal"})
    )
    a_sig = (
        residents.loc[a_mask]
        .groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)["participation_signal"]
        .mean()
        .rename(columns={"participation_signal": "abstainers_mean_participation_signal"})
    )
    p_sig_components = (
        residents.loc[p_mask]
        .groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)
        .agg(
            participants_mean_signal_group_component=("participation_signal_group_component", "mean"),
            participants_mean_signal_fee_component=("participation_signal_fee_component", "mean"),
        )
    )
    a_sig_components = (
        residents.loc[a_mask]
        .groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)
        .agg(
            abstainers_mean_signal_group_component=("participation_signal_group_component", "mean"),
            abstainers_mean_signal_fee_component=("participation_signal_fee_component", "mean"),
        )
    )
    q_update = (
        residents.groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)[
            "participation_q_update_proxy"
        ]
        .mean()
        .rename(columns={"participation_q_update_proxy": "group_mean_participation_q_update_proxy"})
    )
    q_delta = (
        residents.groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)["participation_q_delta"]
        .mean()
        .rename(columns={"participation_q_delta": "group_mean_participation_q_delta"})
    )
    group_p_delta = (
        residents.groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)["participation_p_delta"]
        .mean()
        .rename(columns={"participation_p_delta": "group_mean_participation_p_delta"})
    )
    p_q_update = (
        residents.loc[p_mask]
        .groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)[
            "participation_q_update_proxy"
        ]
        .mean()
        .rename(columns={"participation_q_update_proxy": "participants_mean_participation_q_update_proxy"})
    )
    p_q_delta = (
        residents.loc[p_mask]
        .groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)["participation_q_delta"]
        .mean()
        .rename(columns={"participation_q_delta": "participants_mean_participation_q_delta"})
    )
    p_p_delta = (
        residents.loc[p_mask]
        .groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)["participation_p_delta"]
        .mean()
        .rename(columns={"participation_p_delta": "participants_mean_participation_p_delta"})
    )
    a_q_update = (
        residents.loc[a_mask]
        .groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)[
            "participation_q_update_proxy"
        ]
        .mean()
        .rename(columns={"participation_q_update_proxy": "abstainers_mean_participation_q_update_proxy"})
    )
    a_q_delta = (
        residents.loc[a_mask]
        .groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)["participation_q_delta"]
        .mean()
        .rename(columns={"participation_q_delta": "abstainers_mean_participation_q_delta"})
    )
    a_p_delta = (
        residents.loc[a_mask]
        .groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)["participation_p_delta"]
        .mean()
        .rename(columns={"participation_p_delta": "abstainers_mean_participation_p_delta"})
    )
    learning_diag = (
        residents.groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)
        .agg(
            learning_direction_score=("learning_direction", "mean"),
            learning_intensity_rel_mean=("learning_intensity_rel", "mean"),
            learning_signed_pressure=("learning_signed_pressure", "mean"),
        )
    )
    out = out.merge(reward_counts, on=["step", "area_id", "personality_group_idx"], how="left")
    out = out.merge(participant_reward_counts, on=["step", "area_id", "personality_group_idx"], how="left")
    out = out.merge(abstainer_reward_counts, on=["step", "area_id", "personality_group_idx"], how="left")
    out = out.merge(learning_diag, on=["step", "area_id", "personality_group_idx"], how="left")
    out = out.merge(p_delta_rel, on=["step", "area_id", "personality_group_idx"], how="left")
    out = out.merge(a_delta, on=["step", "area_id", "personality_group_idx"], how="left")
    out = out.merge(p_fee, on=["step", "area_id", "personality_group_idx"], how="left")
    out = out.merge(p_sig, on=["step", "area_id", "personality_group_idx"], how="left")
    out = out.merge(a_sig, on=["step", "area_id", "personality_group_idx"], how="left")
    out = out.merge(p_sig_components, on=["step", "area_id", "personality_group_idx"], how="left")
    out = out.merge(a_sig_components, on=["step", "area_id", "personality_group_idx"], how="left")
    out = out.merge(q_delta, on=["step", "area_id", "personality_group_idx"], how="left")
    out = out.merge(group_p_delta, on=["step", "area_id", "personality_group_idx"], how="left")
    out = out.merge(q_update, on=["step", "area_id", "personality_group_idx"], how="left")
    out = out.merge(p_q_delta, on=["step", "area_id", "personality_group_idx"], how="left")
    out = out.merge(a_q_delta, on=["step", "area_id", "personality_group_idx"], how="left")
    out = out.merge(p_p_delta, on=["step", "area_id", "personality_group_idx"], how="left")
    out = out.merge(a_p_delta, on=["step", "area_id", "personality_group_idx"], how="left")
    out = out.merge(p_q_update, on=["step", "area_id", "personality_group_idx"], how="left")
    out = out.merge(a_q_update, on=["step", "area_id", "personality_group_idx"], how="left")

    # Participation-state switching diagnostics (per area/group/step).
    switch_part = residents[["step", "area_id", "agent_id", "personality_group_idx", "participating"]].copy()
    switch_part = switch_part.sort_values(["area_id", "agent_id", "step"]).reset_index(drop=True)
    switch_part["prev_participating"] = switch_part.groupby(["area_id", "agent_id"], sort=False)["participating"].shift(1)
    comparable_part = switch_part["prev_participating"].isin([True, False])
    switch_part = switch_part.loc[comparable_part].copy()
    if not switch_part.empty:
        switch_part["switch_to_abstain"] = (
            switch_part["prev_participating"].astype(bool)
            & (~switch_part["participating"].astype(bool))
        )
        p_switch_drop = (
            switch_part.groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)[
                "switch_to_abstain"
            ]
            .mean()
            .rename(columns={"switch_to_abstain": "participation_switch_to_abstain_share"})
        )
        out = out.merge(p_switch_drop, on=["step", "area_id", "personality_group_idx"], how="left")

    if not votes.empty and "voted_altruistically" in votes.columns:
        v_mode = votes[["step", "area_id", "agent_id", "voted_altruistically"]].drop_duplicates(
            subset=["step", "area_id", "agent_id"], keep="first"
        ).copy()
        v_mode["step"] = v_mode["step"].astype("int32")
        v_mode["area_id"] = v_mode["area_id"].astype("int32")
        v_mode["agent_id"] = v_mode["agent_id"].astype("int32")
        v_mode = v_mode[v_mode["voted_altruistically"].isin([True, False])]
        if not v_mode.empty:
            v_mode = v_mode.merge(ag, on="agent_id", how="left")
            v_mode = v_mode.merge(
                state[["step", "agent_id", "dissatisfaction_signal"]],
                on=["step", "agent_id"],
                how="left",
            )
            v_mode = v_mode.dropna(subset=["personality_group_idx"])
            v_mode["personality_group_idx"] = v_mode["personality_group_idx"].astype("int16")

            v_alt = (
                v_mode[v_mode["voted_altruistically"] == True]
                .groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)["dissatisfaction_signal"]
                .mean()
                .rename(
                    columns={
                        "dissatisfaction_signal": "altruistic_voters_mean_dissatisfaction_signal"
                    }
                )
            )
            v_non = (
                v_mode[v_mode["voted_altruistically"] == False]
                .groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)["dissatisfaction_signal"]
                .mean()
                .rename(
                    columns={
                        "dissatisfaction_signal": "self_regarding_voters_mean_dissatisfaction_signal"
                    }
                )
            )
            v_mode = v_mode.merge(
                residents[
                    [
                        "step",
                        "area_id",
                        "agent_id",
                        "participation_q_update_proxy",
                        "participation_q_delta",
                        "altruism_update_proxy",
                        "altruism_delta",
                    ]
                ].drop_duplicates(subset=["step", "area_id", "agent_id"], keep="first"),
                on=["step", "area_id", "agent_id"],
                how="left",
            )
            v_alt_q = (
                v_mode[v_mode["voted_altruistically"] == True]
                .groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)[
                    "participation_q_update_proxy"
                ]
                .mean()
                .rename(
                    columns={
                        "participation_q_update_proxy": "altruistic_voters_mean_participation_q_update_proxy"
                    }
                )
            )
            v_non_q = (
                v_mode[v_mode["voted_altruistically"] == False]
                .groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)[
                    "participation_q_update_proxy"
                ]
                .mean()
                .rename(
                    columns={
                        "participation_q_update_proxy": "self_regarding_voters_mean_participation_q_update_proxy"
                    }
                )
            )
            v_alt_q_delta = (
                v_mode[v_mode["voted_altruistically"] == True]
                .groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)["participation_q_delta"]
                .mean()
                .rename(
                    columns={
                        "participation_q_delta": "altruistic_voters_mean_participation_q_delta"
                    }
                )
            )
            v_non_q_delta = (
                v_mode[v_mode["voted_altruistically"] == False]
                .groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)["participation_q_delta"]
                .mean()
                .rename(
                    columns={
                        "participation_q_delta": "self_regarding_voters_mean_participation_q_delta"
                    }
                )
            )
            v_alt_a = (
                v_mode[v_mode["voted_altruistically"] == True]
                .groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)[
                    "altruism_update_proxy"
                ]
                .mean()
                .rename(
                    columns={
                        "altruism_update_proxy": "altruistic_voters_mean_altruism_update_proxy"
                    }
                )
            )
            v_alt_a_delta = (
                v_mode[v_mode["voted_altruistically"] == True]
                .groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)["altruism_delta"]
                .mean()
                .rename(columns={"altruism_delta": "altruistic_voters_mean_altruism_delta"})
            )
            v_non_a_delta = (
                v_mode[v_mode["voted_altruistically"] == False]
                .groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)["altruism_delta"]
                .mean()
                .rename(columns={"altruism_delta": "self_regarding_voters_mean_altruism_delta"})
            )
            v_non_a = (
                v_mode[v_mode["voted_altruistically"] == False]
                .groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)[
                    "altruism_update_proxy"
                ]
                .mean()
                .rename(
                    columns={
                        "altruism_update_proxy": "self_regarding_voters_mean_altruism_update_proxy"
                    }
                )
            )
            # Vote-mode switch share per group/step (among agents with t-1 and t vote-mode observations).
            v_mode = v_mode.sort_values(["area_id", "agent_id", "step"]).reset_index(drop=True)
            v_mode["prev_voted_altruistically"] = v_mode.groupby(["area_id", "agent_id"], sort=False)[
                "voted_altruistically"
            ].shift(1)
            current_counts = (
                v_mode.groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)
                .agg(current_participants=("agent_id", "nunique"))
            )
            comparable = v_mode["prev_voted_altruistically"].isin([True, False])
            switch_block = v_mode.loc[comparable].copy()
            if not switch_block.empty:
                switch_block["switched"] = (
                    switch_block["voted_altruistically"].astype(bool)
                    != switch_block["prev_voted_altruistically"].astype(bool)
                ).astype(int)
                switch_block["switched_from_altruistic"] = (
                    switch_block["prev_voted_altruistically"].astype(bool)
                    & (~switch_block["voted_altruistically"].astype(bool))
                ).astype(int)
                switch_block["switched_from_self_regarding"] = (
                    (~switch_block["prev_voted_altruistically"].astype(bool))
                    & switch_block["voted_altruistically"].astype(bool)
                ).astype(int)
                switch_counts = (
                    switch_block.groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)
                    .agg(
                        switched=("switched", "sum"),
                        switched_from_altruistic=("switched_from_altruistic", "sum"),
                        switched_from_self_regarding=("switched_from_self_regarding", "sum"),
                    )
                )
                v_switch = current_counts.merge(
                    switch_counts,
                    on=["step", "area_id", "personality_group_idx"],
                    how="left",
                ).fillna({"switched": 0, "switched_from_altruistic": 0, "switched_from_self_regarding": 0})
                n = v_switch["current_participants"].to_numpy(dtype=float)
                sw = v_switch["switched"].to_numpy(dtype=float)
                sw_a = v_switch["switched_from_altruistic"].to_numpy(dtype=float)
                sw_n = v_switch["switched_from_self_regarding"].to_numpy(dtype=float)
                v_switch["vote_mode_switch_share"] = np.divide(
                    sw, n, out=np.full_like(sw, np.nan), where=n > 0.0
                )
                v_switch["vote_mode_switch_from_altruistic_share"] = np.divide(
                    sw_a, n, out=np.full_like(sw_a, np.nan), where=n > 0.0
                )
                v_switch["vote_mode_switch_from_self_regarding_share"] = np.divide(
                    sw_n, n, out=np.full_like(sw_n, np.nan), where=n > 0.0
                )
                v_switch = v_switch[
                    [
                        "step",
                        "area_id",
                        "personality_group_idx",
                        "vote_mode_switch_share",
                        "vote_mode_switch_from_altruistic_share",
                        "vote_mode_switch_from_self_regarding_share",
                    ]
                ]
            else:
                v_switch = current_counts.copy()
                v_switch["vote_mode_switch_share"] = np.nan
                v_switch["vote_mode_switch_from_altruistic_share"] = np.nan
                v_switch["vote_mode_switch_from_self_regarding_share"] = np.nan
                v_switch = v_switch[
                    [
                        "step",
                        "area_id",
                        "personality_group_idx",
                        "vote_mode_switch_share",
                        "vote_mode_switch_from_altruistic_share",
                        "vote_mode_switch_from_self_regarding_share",
                    ]
                ]
            out = out.merge(v_alt, on=["step", "area_id", "personality_group_idx"], how="left")
            out = out.merge(v_non, on=["step", "area_id", "personality_group_idx"], how="left")
            out = out.merge(v_alt_q_delta, on=["step", "area_id", "personality_group_idx"], how="left")
            out = out.merge(v_non_q_delta, on=["step", "area_id", "personality_group_idx"], how="left")
            out = out.merge(v_alt_q, on=["step", "area_id", "personality_group_idx"], how="left")
            out = out.merge(v_non_q, on=["step", "area_id", "personality_group_idx"], how="left")
            out = out.merge(v_alt_a_delta, on=["step", "area_id", "personality_group_idx"], how="left")
            out = out.merge(v_non_a_delta, on=["step", "area_id", "personality_group_idx"], how="left")
            out = out.merge(v_alt_a, on=["step", "area_id", "personality_group_idx"], how="left")
            out = out.merge(v_non_a, on=["step", "area_id", "personality_group_idx"], how="left")
            out = out.merge(v_switch, on=["step", "area_id", "personality_group_idx"], how="left")
    for c in (
        "group_reward_count",
        "group_punishment_count",
        "participants_reward_count",
        "participants_punishment_count",
        "abstainers_reward_count",
        "abstainers_punishment_count",
        "learning_direction_score",
        "learning_intensity_rel_mean",
        "learning_signed_pressure",
        "participants_mean_delta_rel",
        "abstainers_mean_delta_rel",
        "participants_mean_fee",
        "participants_mean_fee_over_assets",
        "group_mu_delta_rel",
        "global_mu_delta_rel",
        "group_signal_shrink_weight",
        "group_signal_component",
        "participants_mean_participation_signal",
        "abstainers_mean_participation_signal",
        "participants_mean_signal_group_component",
        "participants_mean_signal_fee_component",
        "abstainers_mean_signal_group_component",
        "abstainers_mean_signal_fee_component",
        "group_mean_participation_q_delta",
        "participants_mean_participation_q_delta",
        "abstainers_mean_participation_q_delta",
        "altruistic_voters_mean_participation_q_delta",
        "self_regarding_voters_mean_participation_q_delta",
        "group_mean_participation_p_delta",
        "participants_mean_participation_p_delta",
        "abstainers_mean_participation_p_delta",
        "group_mean_participation_q_update_proxy",
        "participants_mean_participation_q_update_proxy",
        "abstainers_mean_participation_q_update_proxy",
        "altruistic_voters_mean_participation_q_update_proxy",
        "self_regarding_voters_mean_participation_q_update_proxy",
        "altruistic_voters_mean_dissatisfaction_signal",
        "self_regarding_voters_mean_dissatisfaction_signal",
        "altruistic_voters_mean_altruism_delta",
        "self_regarding_voters_mean_altruism_delta",
        "altruistic_voters_mean_altruism_update_proxy",
        "self_regarding_voters_mean_altruism_update_proxy",
        "vote_mode_switch_share",
        "vote_mode_switch_from_altruistic_share",
        "vote_mode_switch_from_self_regarding_share",
        "participation_switch_to_abstain_share",
        "group_std_q_participation",
        "group_std_participation_probability",
        "group_gini_assets_within",
        "group_gini_dissatisfaction_within",
    ):
        if c not in out.columns:
            out[c] = np.nan
        if c in out.columns:
            out[c] = out[c].astype("float32")

    out = out.rename(columns={"personality_group_idx": "group_idx"})
    out["group_idx"] = out["group_idx"].astype("int16")
    out["mean_assets"] = out["mean_assets"].astype("float32")
    out["mean_dissatisfaction"] = out["mean_dissatisfaction"].astype("float32")
    for c in (
        "group_reward_count",
        "group_punishment_count",
        "participants_reward_count",
        "participants_punishment_count",
        "abstainers_reward_count",
        "abstainers_punishment_count",
    ):
        if c not in out.columns:
            out[c] = 0
        out[c] = out[c].fillna(0).astype("int32")
    return out[cols].sort_values(["area_id", "step", "group_idx"]).reset_index(drop=True)

def _weighted_dist_to_reality_by_step(*, area_steps: pd.DataFrame, step_index: np.ndarray) -> list[float]:
    return _weighted_metric_by_step(area_steps=area_steps, step_index=step_index, value_col="dist_to_reality")

def _weighted_metric_by_step(*, area_steps: pd.DataFrame, step_index: np.ndarray, value_col: str) -> list[float]:
    out: list[float] = []
    if value_col not in area_steps.columns:
        return [float("nan") for _ in step_index]
    grouped = area_steps.groupby("step", sort=True)[[value_col, "eligible_voters"]]
    for step in step_index:
        if int(step) not in grouped.groups:
            out.append(float("nan"))
            continue
        block = grouped.get_group(int(step))
        weights = block["eligible_voters"].to_numpy(dtype=float)
        vals = block[value_col].to_numpy(dtype=float)
        denom = float(np.sum(weights))
        if denom <= 0.0:
            out.append(float("nan"))
        else:
            out.append(float(np.sum(vals * weights) / denom))
    return out

def _weighted_vector_metric_by_step(
    *,
    area_steps: pd.DataFrame,
    step_index: np.ndarray,
    value_cols: list[str],
) -> np.ndarray:
    if not value_cols:
        return np.empty((len(step_index), 0), dtype=np.float64)
    missing = [c for c in value_cols if c not in area_steps.columns]
    if missing:
        return np.full((len(step_index), len(value_cols)), np.nan, dtype=np.float64)
    out = np.full((len(step_index), len(value_cols)), np.nan, dtype=np.float64)
    grouped = area_steps.groupby("step", sort=True)[value_cols + ["eligible_voters"]]
    for idx, step in enumerate(step_index):
        if int(step) not in grouped.groups:
            continue
        block = grouped.get_group(int(step))
        w = block["eligible_voters"].to_numpy(dtype=float)
        denom = float(np.sum(w))
        if denom <= 0.0:
            continue
        vals = block[value_cols].to_numpy(dtype=float)
        out[idx, :] = np.sum(vals * w[:, None], axis=0) / denom
    return out

def _diversity_entropy_by_step(*, votes: pd.DataFrame, step_index: np.ndarray, num_options: int) -> list[float]:
    out: list[float] = []
    if num_options <= 1:
        return [float(0.0) for _ in step_index]
    norm = log(float(num_options))
    if norm <= 0.0:
        return [float(0.0) for _ in step_index]

    grouped = votes.groupby("step", sort=True)["rank_1_option_id"] if not votes.empty else None
    for step in step_index:
        if grouped is None or int(step) not in grouped.groups:
            out.append(float("nan"))
            continue
        s = grouped.get_group(int(step)).dropna()
        if s.empty:
            out.append(float("nan"))
            continue
        counts = s.value_counts().to_numpy(dtype=float)
        probs = counts / float(np.sum(counts))
        entropy = float(-np.sum(probs * np.log(probs + 1e-15)))
        out.append(float(max(0.0, min(1.0, entropy / norm))))
    return out

def _diversity_entropy_by_step_area(*, votes: pd.DataFrame, num_options: int) -> pd.DataFrame:
    if votes.empty:
        return pd.DataFrame(columns=["step", "area_id", "diversity_first_choice_entropy"])
    if num_options <= 1:
        tmp = votes[["step", "area_id"]].drop_duplicates()
        tmp["diversity_first_choice_entropy"] = 0.0
        return tmp
    norm = log(float(num_options))
    if norm <= 0.0:
        tmp = votes[["step", "area_id"]].drop_duplicates()
        tmp["diversity_first_choice_entropy"] = 0.0
        return tmp

    rows: list[dict[str, Any]] = []
    for (step, area_id), block in votes.groupby(["step", "area_id"], sort=True):
        s = block["rank_1_option_id"].dropna()
        if s.empty:
            val = float("nan")
        else:
            counts = s.value_counts().to_numpy(dtype=float)
            probs = counts / float(np.sum(counts))
            entropy = float(-np.sum(probs * np.log(probs + 1e-15)))
            val = float(max(0.0, min(1.0, entropy / norm)))
        rows.append(
            {
                "step": int(step),
                "area_id": int(area_id),
                "diversity_first_choice_entropy": val,
            }
        )
    return pd.DataFrame(rows)

def _personal_dists_from_static(*, static: dict[str, Any], num_colors: int) -> np.ndarray:
    pod = static.get("personal_opt_dist")
    if not isinstance(pod, dict) or len(pod) == 0:
        return np.asarray([], dtype=np.float64)
    rows: list[np.ndarray] = []
    for _agent_id, v in pod.items():
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 1 or arr.size != int(num_colors):
            continue
        s = float(np.sum(arr))
        if s <= 0.0:
            continue
        rows.append(arr / s)
    if not rows:
        return np.asarray([], dtype=np.float64)
    return np.vstack(rows).astype(np.float64)

def _summary_ordering_distance_func(meta: dict[str, Any]):
    run_meta = (meta.get("run", {}) or {}) if isinstance(meta, dict) else {}
    name = str(run_meta.get("distance_impl_name", "") or "").strip()
    if name in {"kendall_tau_order", "kendall_tau"}:
        return kendall_tau_order
    return spearman_fr_order

def _summary_ordering_from_distribution_tie_aware(
    dist: np.ndarray,
    *,
    reference_ordering: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    atol: float = 1e-9,
    rtol: float = 1e-8,
) -> np.ndarray:
    """Compatibility wrapper; canonical implementation lives in utils.representations."""
    return distribution_to_ordering_tie_aware(
        dist,
        reference_ordering=reference_ordering,
        rng=rng,
        atol=atol,
        rtol=rtol,
    )
