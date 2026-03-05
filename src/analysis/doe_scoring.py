from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import itertools

import numpy as np
import pandas as pd
import yaml

from src.utils.ballots import score_options_c2
from src.utils.distance_functions import kendall_tau_order, spearman_fr_order
from src.utils.social_welfare_functions import (
    approval_voting,
    borda_rule,
    majority_rule,
    random_rule,
    utilitarian_rule,
)
from src.utils.representations import distribution_to_ordering_tie_aware


DEFAULT_SCORING_THRESHOLDS: dict[str, float] = {
    "max_all_abstain_stretch": 10.0,
    "min_winner_changes_post_burnin": 3.0,
    "min_winner_change_rate_post_burnin": 0.01,
    "max_winner_change_rate_post_burnin": 0.80,
    "min_group_turnout_range_mean": 0.03,
    "min_roll3_group_turnout_range_max": 0.3,
    "min_roll20_group_turnout_range_max": 0.1,
    "min_turnout_std": 0.5,
    "min_gini_std": 1.0,
    "min_dist_std": 0.02,
    "min_winner_entropy_norm": 0.25,
    "min_dist_nonzero_share": 0.05,
    "min_competitive_step_share": 0.05,
    "min_mean_turnout": 20.0,
    "max_mean_turnout": 90.0,
    "puzzle_conflict_min_dist": 0.33,
    "min_puzzle_conflict_step_share_for_gate": 0.20,
    "max_puzzle_dominance_share_conflict": 0.95,
    "min_power_recovery_share_conflict": 0.02,
    "puzzle_dominance_share_score_low": 0.65,
    "puzzle_dominance_share_score_high": 0.90,
    # Explicit lock-in recovery sequence detection (conflict-only):
    # power lock-in episode (same dominant participant group + power-dominant margin)
    # -> dominant-group altruism surge
    # -> puzzle takes over (margin crosses positive) in forward window.
    "lockin_episode_min_len_steps": 8.0,
    "lockin_recovery_window_steps": 40.0,
    "lockin_min_dominant_participant_share": 0.55,
    "lockin_min_altruism_share": 0.80,
    "lockin_min_altruism_lift": 0.12,
    "lockin_margin_power_threshold": 0.0,
    "lockin_margin_puzzle_recovery_threshold": 0.0,
    # Soft-score shaping for lock-in recovery share.
    "lockin_recovery_share_score_zero_at": 0.05,
    "lockin_recovery_share_score_good_min": 0.40,
    # Moderate turnout-share cannibalization-and-rebound recovery signal.
    # Phase-1 (A eats B) is stricter; Phase-2 (B eats back) has slightly lower bars.
    "moderate_min_group_size": 6.0,
    "moderate_takeover_loss_frac_min": 0.50,
    "moderate_takeover_capture_frac_min": 0.50,
    "moderate_rebound_loss_frac_min": 0.35,
    "moderate_rebound_share_vs_a_peak_min": 0.40,
    "moderate_min_steps": 5.0,
    "moderate_min_peak_share": 0.00,
    "moderate_length_bonus_max": 0.15,
    "moderate_altruism_bonus_weight": 0.0,
    "moderate_altruism_bonus_floor": 0.60,
    # Quality gate for the (light) moderate selector bonus.
    "moderate_selector_quality_gate_zero_at": 0.58,
    "moderate_selector_quality_gate_good_min": 0.72,
    # Turnout shape soft score (step-count independent; avoids deceptive mean-only scoring).
    "turnout_start_score_low": 30.0,
    "turnout_start_score_high": 70.0,
    "turnout_end_score_low": 20.0,
    "turnout_end_score_high": 80.0,
    "turnout_drop_score_good_max": 15.0,
    "turnout_drop_score_zero_at": 60.0,
    "turnout_decline_score_good_max": 0.08,
    "turnout_decline_score_zero_at": 0.50,
    "turnout_outside_band_share_good_max": 0.20,
    "turnout_outside_band_share_zero_at": 0.80,
    # Participation learning stabilization soft scores (exact q deltas from q snapshots where available).
    "participation_q_delta_mean_abs_good_max": 0.0015,
    "participation_q_delta_mean_abs_zero_at": 0.0120,
    "participation_q_delta_late_mean_abs_good_max": 0.0010,
    "participation_q_delta_late_mean_abs_zero_at": 0.0080,
}

DEFAULT_SCORING_WEIGHTS: dict[str, float] = {
    "quality_mean": 0.67,
    "discriminability": 0.0,
    "seed_robustness": 0.30,
    "moderate_recovery_selector": 0.03,
}

DEFAULT_STAGE_WEIGHTS: dict[str, float] = {
    "viability": 0.60,
    "quality_bundle": 0.40,
}


REQUIRED_HARD_GATE_COLUMNS: tuple[str, ...] = (
    "max_all_abstain_stretch",
    "winner_changes_post_burnin",
    "winner_change_rate_post_burnin",
    "group_turnout_range_mean",
    "roll3_group_turnout_range_max",
    "roll20_group_turnout_range_max",
    "winner_entropy_norm",
    "dist_nonzero_share",
    "competitive_step_share",
    "mean_turnout",
    "turnout_std",
    "gini_std",
    "dist_std",
)

REQUIRED_PRIMARY_SCORING_COLUMNS: tuple[str, ...] = (
    "design_id",
    "seed",
    "passes_hard_gates",
    "turnout_std",
    "gini_std",
    "dist_std",
    "group_participation_std",
    "group_turnout_range_mean",
    "roll3_group_turnout_range_mean",
    "roll3_group_turnout_range_max",
    "roll20_group_turnout_range_mean",
    "roll20_group_turnout_range_max",
    "group_turnout_residual_abs_mean",
    "participant_abstainer_delta_rel_gap_abs",
    "group_participant_abstainer_delta_rel_gap_abs",
    "winner_entropy_norm",
    "dist_nonzero_share",
    "competitive_step_share",
    "winner_change_rate_post_burnin",
    "mean_turnout",
    "turnout_start_window_mean",
    "turnout_end_window_mean",
    "turnout_drop_start_end",
    "turnout_decline_slope_norm",
    "turnout_outside_20_80_share",
)

OPTIONAL_PRIMARY_SCORING_COLUMNS: tuple[str, ...] = (
    "participation_q_delta_mean_abs",
    "participation_q_delta_late_window_mean_abs",
    "participation_q_delta_group_dispersion_late_w",
    "participant_share_max_abs_drift_20",
    "participant_share_mean_abs_drift_20_w",
    "participant_share_turnover_rate_w",
    "puzzle_dominance_share_conflict",
    "lockin_recovery_share_conflict",
    "moderate_recovery_strength_run",
    "moderate_recovery_event_count",
    "moderate_recovery_pair_coverage",
)

PRIMARY_QUALITY_COMPONENT_KEYS: tuple[str, ...] = (
    "z_turnout_std",
    "z_gini_std",
    "z_dist_std",
    "z_group_std",
    "z_group_turnout_range",
    "z_roll3_group_turnout_range",
    "z_roll3_group_turnout_range_max",
    "z_roll20_group_turnout_range",
    "z_roll20_group_turnout_range_max",
    "z_group_turnout_resid",
    "z_pa_gap",
    "z_group_pa_gap",
    "z_winner_entropy",
    "z_dist_nonzero_share",
    "z_competitive_step_share",
    "z_winner_changes",
    "z_turnout_shape",
    "z_q_delta_mean_stability",
    "z_q_delta_late_stability",
    "z_turnout_drop_stability",
    "z_turnout_decline_stability",
    "z_q_delta_group_dispersion_late",
    "z_participant_composition_dynamics",
    "z_puzzle_dom_balance_conflict",
)

REQUIRED_ROBUST_SCORING_COLUMNS: tuple[str, ...] = (
    "design_id",
    "seed",
    "mean_turnout",
    "mean_gini",
    "mean_dist",
    "group_participation_std",
)


def _require_columns(df: pd.DataFrame, required: tuple[str, ...], *, context: str) -> None:
    missing = sorted(col for col in required if col not in df.columns)
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def _require_numeric_notna(df: pd.DataFrame, required: tuple[str, ...], *, context: str) -> None:
    bad: dict[str, int] = {}
    for col in required:
        s = pd.to_numeric(df[col], errors="coerce")
        n_bad = int(s.isna().sum())
        if n_bad > 0:
            bad[col] = n_bad
    if bad:
        details = ", ".join(f"{col}: {count}" for col, count in sorted(bad.items()))
        raise ValueError(f"{context} has NA/non-numeric values in required columns: {details}")


def load_selection_objective(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Selection objective config must be a JSON object.")
    allowed = {
        "version",
        "thresholds",
        "weights",
        "stage_weights",
        "strict_completeness",
        "quality_component_weights",
    }
    unknown = sorted(set(payload.keys()) - allowed)
    if unknown:
        raise ValueError(f"Unknown selection objective keys: {unknown}")

    thr = dict(DEFAULT_SCORING_THRESHOLDS)
    obj_thr = payload.get("thresholds", {})
    if obj_thr is not None:
        if not isinstance(obj_thr, dict):
            raise ValueError("thresholds must be a JSON object.")
        for k, v in obj_thr.items():
            # Backward-compat migration for old absolute chaos gate key.
            if k == "max_winner_changes_post_burnin":
                # ignored in favor of rate-based gate, but accepted for old files
                continue
            if k not in thr:
                raise ValueError(f"Unknown threshold key: {k}")
            thr[k] = float(v)

    w = dict(DEFAULT_SCORING_WEIGHTS)
    obj_w = payload.get("weights", {})
    if obj_w is not None:
        if not isinstance(obj_w, dict):
            raise ValueError("weights must be a JSON object.")
        for k, v in obj_w.items():
            if k not in w:
                raise ValueError(f"Unknown weight key: {k}")
            w[k] = float(v)

    sw = dict(DEFAULT_STAGE_WEIGHTS)
    obj_sw = payload.get("stage_weights", {})
    if obj_sw is not None:
        if not isinstance(obj_sw, dict):
            raise ValueError("stage_weights must be a JSON object.")
        for k, v in obj_sw.items():
            if k not in sw:
                raise ValueError(f"Unknown stage_weight key: {k}")
            sw[k] = float(v)

    strict = payload.get("strict_completeness", True)
    if not isinstance(strict, bool):
        raise ValueError("strict_completeness must be a boolean.")

    qcw_payload = payload.get("quality_component_weights", None)
    qcw: dict[str, float] | None = None
    if qcw_payload is not None:
        if not isinstance(qcw_payload, dict):
            raise ValueError("quality_component_weights must be a JSON object.")
        qcw_raw: dict[str, float] = {}
        for k, v in qcw_payload.items():
            if k not in PRIMARY_QUALITY_COMPONENT_KEYS:
                raise ValueError(f"Unknown quality_component_weights key: {k}")
            qcw_raw[k] = float(v)
        pos_total = float(sum(max(0.0, x) for x in qcw_raw.values()))
        if pos_total <= 0.0:
            raise ValueError("quality_component_weights must have positive total weight.")
        qcw = {
            k: (max(0.0, qcw_raw.get(k, 0.0)) / pos_total)
            for k in PRIMARY_QUALITY_COMPONENT_KEYS
        }

    return {
        "path": str(p),
        "version": payload.get("version"),
        "thresholds": thr,
        "weights": w,
        "stage_weights": sw,
        "strict_completeness": bool(strict),
        "quality_component_weights": qcw,
    }


def _safe_std(series: pd.Series) -> float:
    if len(series) <= 1:
        return 0.0
    v = float(series.std())
    if not np.isfinite(v):
        return 0.0
    return v


def _mean_finite_pairwise_corr(window: np.ndarray) -> float | None:
    """Return mean pairwise correlation across columns, excluding constant columns."""
    if window.ndim != 2 or window.shape[0] < 2 or window.shape[1] < 2:
        return None
    if not np.isfinite(window).all():
        return None
    std = np.std(window, axis=0)
    keep = np.isfinite(std) & (std > 1e-12)
    if int(np.count_nonzero(keep)) < 2:
        return None
    w = window[:, keep]
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.corrcoef(w, rowvar=False)
    if c.ndim != 2 or c.shape[0] < 2:
        return None
    tri = c[np.triu_indices_from(c, k=1)]
    tri = tri[np.isfinite(tri)]
    if len(tri) == 0:
        return None
    return float(np.mean(tri))


def _max_true_stretch(mask: np.ndarray) -> int:
    best = 0
    cur = 0
    for v in mask:
        if bool(v):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _first_true_streak(mask: np.ndarray, min_len: int) -> tuple[int, int] | None:
    cur = 0
    start = 0
    for i, v in enumerate(mask):
        if bool(v):
            if cur == 0:
                start = i
            cur += 1
            if cur >= int(min_len):
                return int(start), int(i)
        else:
            cur = 0
    return None


def _winner_changes(win_ids: np.ndarray) -> int:
    if len(win_ids) <= 1:
        return 0
    return int(np.sum(win_ids[1:] != win_ids[:-1]))


def _winner_entropy_norm(win_ids: np.ndarray) -> float:
    if len(win_ids) == 0:
        return 0.0
    vals, counts = np.unique(win_ids, return_counts=True)
    if len(vals) <= 1:
        return 0.0
    p = counts.astype(float) / float(np.sum(counts))
    ent = -float(np.sum(p * np.log(p + 1e-12)))
    max_ent = float(np.log(len(vals)))
    if max_ent <= 0.0 or not np.isfinite(ent):
        return 0.0
    return float(np.clip(ent / max_ent, 0.0, 1.0))


def _norm01(series: pd.Series, *, higher_better: bool = True) -> pd.Series:
    s = series.astype(float)
    lo = float(s.min())
    hi = float(s.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
        z = pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    else:
        z = (s - lo) / (hi - lo)
    if not higher_better:
        z = 1.0 - z
    return z.clip(0.0, 1.0)


def _band_pref01(series: pd.Series, *, low: float, high: float) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if high < low:
        low, high = high, low
    mask = s.notna()
    if not mask.any():
        return out
    vals = s[mask]
    score = pd.Series(np.ones(len(vals), dtype=float), index=vals.index)
    width = float(high - low)
    if not np.isfinite(width) or width <= 1e-12:
        width = 1.0
    below = vals < low
    if bool(below.any()):
        score.loc[below] = (1.0 - ((low - vals.loc[below]) / width)).clip(0.0, 1.0)
    above = vals > high
    if bool(above.any()):
        score.loc[above] = (1.0 - ((vals.loc[above] - high) / width)).clip(0.0, 1.0)
    out.loc[mask] = score.clip(0.0, 1.0)
    return out


def _upper_bound_pref01(series: pd.Series, *, good_max: float, zero_at: float) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if zero_at < good_max:
        good_max, zero_at = zero_at, good_max
    mask = s.notna()
    if not mask.any():
        return out
    vals = s[mask]
    score = pd.Series(np.ones(len(vals), dtype=float), index=vals.index)
    if zero_at <= good_max + 1e-12:
        score.loc[vals > good_max] = 0.0
        out.loc[mask] = score
        return out
    mid = vals > good_max
    if bool(mid.any()):
        score.loc[mid] = (1.0 - ((vals.loc[mid] - good_max) / (zero_at - good_max))).clip(0.0, 1.0)
    out.loc[mask] = score.clip(0.0, 1.0)
    return out


def _lower_bound_pref01(series: pd.Series, *, zero_at: float, good_min: float) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if good_min < zero_at:
        zero_at, good_min = good_min, zero_at
    mask = s.notna()
    if not mask.any():
        return out
    vals = s[mask]
    score = pd.Series(np.zeros(len(vals), dtype=float), index=vals.index)
    if good_min <= zero_at + 1e-12:
        score.loc[vals >= good_min] = 1.0
        out.loc[mask] = score
        return out
    above = vals >= good_min
    if bool(above.any()):
        score.loc[above] = 1.0
    mid = (vals > zero_at) & (vals < good_min)
    if bool(mid.any()):
        score.loc[mid] = ((vals.loc[mid] - zero_at) / (good_min - zero_at)).clip(0.0, 1.0)
    out.loc[mask] = score.clip(0.0, 1.0)
    return out


def _ordering_distance_func_from_meta(meta: dict[str, Any]):
    run_meta = (meta.get("run", {}) or {}) if isinstance(meta, dict) else {}
    name = str(run_meta.get("distance_impl_name", "") or "").strip()
    if name in {"kendall_tau_order", "kendall_tau"}:
        return kendall_tau_order
    return spearman_fr_order


def _current_rule_power_ordering_for_run(
    *,
    meta: dict[str, Any],
    static: dict[str, Any],
    agents: pd.DataFrame,
    num_colors: int,
) -> np.ndarray | None:
    run_meta = (meta.get("run", {}) or {}) if isinstance(meta, dict) else {}
    rule_idx = int(run_meta.get("rule_idx", -1))
    if rule_idx not in {0, 1, 2, 3, 4}:
        return None
    pgi = (static.get("personality_group_info") or {}) if isinstance(static, dict) else {}
    personality_groups = np.asarray(pgi.get("personality_groups", []), dtype=np.int64)
    if personality_groups.ndim != 2 or personality_groups.shape[0] <= 0:
        return None
    n_groups = int(personality_groups.shape[0])
    step0 = int(agents["step"].min()) if "step" in agents.columns and len(agents) > 0 else None
    if step0 is None:
        return None
    agents0 = agents.loc[agents["step"].astype(int) == step0].copy()
    if agents0.empty or "personality_group_idx" not in agents0.columns:
        return None
    counts = (
        agents0.groupby("personality_group_idx", as_index=False)
        .size()
        .rename(columns={"size": "residents"})
        .sort_values("personality_group_idx")
    )
    residents_by_group = np.zeros(n_groups, dtype=int)
    for _, row in counts.iterrows():
        gi = int(row["personality_group_idx"])
        if 0 <= gi < n_groups:
            residents_by_group[gi] = int(row["residents"])
    if int(residents_by_group.sum()) <= 0:
        return None

    options = np.asarray(list(itertools.permutations(range(int(num_colors)))), dtype=np.int64)
    dist_func = _ordering_distance_func_from_meta(meta)
    search_pairs = list(itertools.combinations(range(int(num_colors)), 2))
    pref_rows: list[np.ndarray] = []
    for gi in range(n_groups):
        cnt = int(residents_by_group[gi])
        if cnt <= 0:
            continue
        target = np.asarray(personality_groups[gi][:num_colors], dtype=np.int64)
        scores = score_options_c2(
            target_ordering=target,
            options=options,
            distance_func=dist_func,
            color_search_pairs=search_pairs,
        ).astype(np.float32)
        pref_rows.append(np.repeat(scores[None, :], cnt, axis=0))
    if not pref_rows:
        return None
    pref_table = np.vstack(pref_rows)
    rule_fns = [majority_rule, approval_voting, utilitarian_rule, borda_rule, random_rule]
    fn = rule_fns[rule_idx]
    seed = int(run_meta.get("run_seed", 0))
    rng = np.random.default_rng((seed * 1_000_003 + 97 * (rule_idx + 1)) % (2**63 - 1))
    try:
        opt_order = np.asarray(fn(pref_table, rng=rng), dtype=np.int64)
        if opt_order.size <= 0:
            return None
        winning_option_id = int(opt_order[0])
        if not (0 <= winning_option_id < int(options.shape[0])):
            return None
        return np.asarray(options[winning_option_id], dtype=np.int64)
    except Exception:
        return None


def _compute_puzzle_power_metrics_for_run(
    *,
    run_dir: Path,
    area_steps: pd.DataFrame,
    agents: pd.DataFrame,
    burn_in_steps: int,
    conflict_min_dist: float,
) -> dict[str, float]:
    step_tbl = _compute_puzzle_power_step_table(
        run_dir=run_dir,
        area_steps=area_steps,
        agents=agents,
        burn_in_steps=burn_in_steps,
        conflict_min_dist=conflict_min_dist,
    )
    out = {
        "puzzle_conflict_step_share": np.nan,
        "puzzle_dominance_share_conflict": np.nan,
        "power_recovery_share_conflict": np.nan,
        "puzzle_power_margin_mean_conflict": np.nan,
    }
    if step_tbl is None or len(step_tbl) == 0:
        return out
    conflict = step_tbl["conflict"].to_numpy(dtype=bool)
    out["puzzle_conflict_step_share"] = float(np.mean(conflict)) if conflict.size else np.nan
    if not conflict.any():
        return out
    margin = pd.to_numeric(step_tbl.loc[conflict, "margin"], errors="coerce").to_numpy(dtype=float)
    margin = margin[np.isfinite(margin)]
    if margin.size == 0:
        return out
    out["puzzle_dominance_share_conflict"] = float(np.mean(margin > 0.0))
    out["power_recovery_share_conflict"] = float(np.mean(margin < 0.0))
    out["puzzle_power_margin_mean_conflict"] = float(np.mean(margin))
    return out


def _compute_puzzle_power_step_table(
    *,
    run_dir: Path,
    area_steps: pd.DataFrame,
    agents: pd.DataFrame,
    burn_in_steps: int,
    conflict_min_dist: float,
) -> pd.DataFrame | None:
    if "puzzle_distance" not in area_steps.columns:
        return None
    puzzle_cols = [c for c in area_steps.columns if c.startswith("puzzle_color_")]
    if not puzzle_cols:
        return None
    try:
        meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
        static = json.loads((run_dir / "static.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    run_meta = (meta.get("run", {}) or {}) if isinstance(meta, dict) else {}
    if str(run_meta.get("quality_target_mode", "reality")) != "puzzle":
        return None
    num_colors = int(static.get("num_colors", len(puzzle_cols)) or len(puzzle_cols))
    if num_colors <= 1:
        return None
    if "winning_option_id" not in area_steps.columns:
        return None
    has_puzzle_ids = "puzzle_ordering_id" in area_steps.columns
    need_pcols = [f"puzzle_color_{i}" for i in range(num_colors)]
    if not has_puzzle_ids and not all(c in area_steps.columns for c in need_pcols):
        return None
    power_ord = _current_rule_power_ordering_for_run(meta=meta, static=static, agents=agents, num_colors=num_colors)
    if power_ord is None:
        return None

    dist_func = _ordering_distance_func_from_meta(meta)
    search_pairs = list(itertools.combinations(range(int(num_colors)), 2))
    options = np.asarray(list(itertools.permutations(range(int(num_colors)))), dtype=np.int64)
    agg: dict[str, tuple[str, str]] = {
        "winning_option_id": ("winning_option_id", "first"),
        "puzzle_distance": ("puzzle_distance", "mean"),
    }
    if has_puzzle_ids:
        agg["puzzle_ordering_id"] = ("puzzle_ordering_id", "first")
    for c in need_pcols:
        agg[c] = (c, "mean")
    df = (
        area_steps.groupby("step", as_index=False)
        .agg(**agg)
        .sort_values("step")
        .reset_index(drop=True)
    )
    df = df.loc[pd.to_numeric(df["step"], errors="coerce").astype(float) > float(burn_in_steps)].copy()
    if df.empty:
        return None
    tie_rng = np.random.default_rng(int(run_meta.get("run_seed", 0)) + 31337)
    prev_pord: np.ndarray | None = None
    rows: list[dict[str, float | int | bool]] = []
    for row in df.itertuples(index=False):
        step = int(getattr(row, "step", -1))
        oid = int(getattr(row, "winning_option_id", -1))
        if step < 0 or not (0 <= oid < int(options.shape[0])):
            continue
        out_ord = np.asarray(options[oid], dtype=np.int64)
        pord: np.ndarray | None = None
        if has_puzzle_ids:
            pid = int(getattr(row, "puzzle_ordering_id", -1))
            if 0 <= pid < int(options.shape[0]):
                pord = np.asarray(options[pid], dtype=np.int64)
        if pord is None:
            pvals = np.asarray([getattr(row, c) for c in need_pcols], dtype=float)
            if not np.isfinite(pvals).all():
                continue
            pord = distribution_to_ordering_tie_aware(
                pvals,
                reference_ordering=prev_pord,
                rng=tie_rng,
                atol=1e-9,
                rtol=1e-8,
            )
        prev_pord = pord
        # Reuse logged puzzle_distance if finite; otherwise recompute.
        pd_logged = float(getattr(row, "puzzle_distance", np.nan))
        if np.isfinite(pd_logged):
            d_opuz = pd_logged
        else:
            d_opuz = float(dist_func(out_ord, pord, search_pairs))
        d_opow = float(dist_func(out_ord, power_ord, search_pairs))
        d_ppow = float(dist_func(pord, power_ord, search_pairs))
        if np.isfinite(d_opuz) and np.isfinite(d_opow) and np.isfinite(d_ppow):
            rows.append(
                {
                    "step": int(step),
                    "winning_option_id": int(oid),
                    "d_out_puz": float(d_opuz),
                    "d_out_pow": float(d_opow),
                    "d_puz_pow": float(d_ppow),
                    "conflict": bool(d_ppow >= float(conflict_min_dist)),
                    "margin": float(d_opow - d_opuz),
                }
            )
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


def _compute_lockin_recovery_metrics_for_run(
    *,
    run_dir: Path,
    area_steps: pd.DataFrame,
    agents: pd.DataFrame,
    burn_in_steps: int,
    conflict_min_dist: float,
    episode_min_len_steps: int,
    recovery_window_steps: int,
    min_dominant_participant_share: float,
    min_altruism_share: float,
    min_altruism_lift: float,
    margin_power_threshold: float,
    margin_puzzle_recovery_threshold: float,
) -> dict[str, float]:
    out = {
        "lockin_episode_count_conflict": 0.0,
        "lockin_altruism_surge_count_conflict": 0.0,
        "lockin_recovery_event_count_conflict": 0.0,
        "lockin_recovery_share_conflict": np.nan,
        "lockin_altruism_surge_share_conflict": np.nan,
        "lockin_recovery_lag_mean_steps_conflict": np.nan,
    }
    step_tbl = _compute_puzzle_power_step_table(
        run_dir=run_dir,
        area_steps=area_steps,
        agents=agents,
        burn_in_steps=burn_in_steps,
        conflict_min_dist=conflict_min_dist,
    )
    if step_tbl is None or len(step_tbl) == 0:
        return out
    steps_sorted = sorted(pd.to_numeric(step_tbl["step"], errors="coerce").dropna().astype(int).unique().tolist())
    if not steps_sorted:
        return out

    # Dominant participant group by step.
    ap = agents.copy()
    if "eligible_for_election" in ap.columns:
        ap = ap.loc[ap["eligible_for_election"] == True]
    req = {"step", "personality_group_idx", "participating"}
    if not req <= set(ap.columns):
        return out
    gp = (
        ap.groupby(["step", "personality_group_idx"], as_index=False)
        .agg(participants=("participating", "sum"))
        .sort_values(["step", "participants", "personality_group_idx"], ascending=[True, False, True])
    )
    if len(gp) == 0:
        return out
    totals = gp.groupby("step", as_index=False).agg(total_participants=("participants", "sum"))
    top = gp.groupby("step", as_index=False).first()
    dom = top.merge(totals, on="step", how="left")
    dom["dominant_participant_share"] = np.where(
        pd.to_numeric(dom["total_participants"], errors="coerce").to_numpy(dtype=float) > 0.0,
        pd.to_numeric(dom["participants"], errors="coerce").to_numpy(dtype=float)
        / pd.to_numeric(dom["total_participants"], errors="coerce").to_numpy(dtype=float),
        np.nan,
    )
    dom = dom.rename(columns={"personality_group_idx": "dominant_group_idx"})

    seq = step_tbl.merge(dom.loc[:, ["step", "dominant_group_idx", "dominant_participant_share"]], on="step", how="left")
    seq["dominant_group_idx"] = pd.to_numeric(seq["dominant_group_idx"], errors="coerce")
    if len(seq) == 0:
        return out
    seq = seq.sort_values("step").reset_index(drop=True)
    step_rows = {int(r["step"]): r for _, r in seq.iterrows()}

    cond_vals: list[tuple[int, int]] = []  # (step, dominant_group_idx)
    for step in steps_sorted:
        r = step_rows.get(int(step))
        if r is None:
            continue
        conflict = bool(r.get("conflict", False))
        margin = float(r.get("margin", np.nan))
        dom_share = float(r.get("dominant_participant_share", np.nan))
        dom_group = r.get("dominant_group_idx", np.nan)
        if (not conflict) or (not np.isfinite(margin)) or (margin >= float(margin_power_threshold)):
            continue
        if (not np.isfinite(dom_share)) or (dom_share < float(min_dominant_participant_share)):
            continue
        if not np.isfinite(dom_group):
            continue
        cond_vals.append((int(step), int(dom_group)))

    if not cond_vals:
        return out

    # Segment into episodes: consecutive steps with same dominant group.
    episodes: list[tuple[int, int, int]] = []  # (start_step, end_step, dominant_group_idx)
    seg_start, seg_prev, seg_group = cond_vals[0][0], cond_vals[0][0], cond_vals[0][1]
    for step, grp in cond_vals[1:]:
        if (step == seg_prev + 1) and (grp == seg_group):
            seg_prev = step
            continue
        if (seg_prev - seg_start + 1) >= int(episode_min_len_steps):
            episodes.append((seg_start, seg_prev, seg_group))
        seg_start, seg_prev, seg_group = step, step, grp
    if (seg_prev - seg_start + 1) >= int(episode_min_len_steps):
        episodes.append((seg_start, seg_prev, seg_group))

    n_episode = int(len(episodes))
    if n_episode <= 0:
        return out

    # Dominant-group altruistic vote share by step (from vote logs), loaded only
    # for runs that actually contain lock-in episodes.
    votes_path = run_dir / "votes.parquet"
    if not votes_path.exists():
        return out
    try:
        votes = pd.read_parquet(votes_path)
    except Exception:
        return out
    if not {"step", "agent_id", "voted_altruistically"} <= set(votes.columns):
        return out
    group_map = ap.loc[:, ["step", "agent_id", "personality_group_idx"]].drop_duplicates(
        subset=["step", "agent_id"]
    )
    vm = votes.merge(group_map, on=["step", "agent_id"], how="left")
    vm = vm.loc[vm["personality_group_idx"].notna()].copy()
    if len(vm) == 0:
        return out
    vm["voted_altruistically"] = pd.to_numeric(vm["voted_altruistically"], errors="coerce")
    altru = (
        vm.groupby(["step", "personality_group_idx"], as_index=False)
        .agg(group_altruism_share=("voted_altruistically", "mean"))
    )
    altru_lookup: dict[tuple[int, int], float] = {}
    for row in altru.itertuples(index=False):
        st = int(getattr(row, "step"))
        gi = int(getattr(row, "personality_group_idx"))
        av = float(getattr(row, "group_altruism_share"))
        if np.isfinite(av):
            altru_lookup[(st, gi)] = av

    n_surge = 0
    n_recovery = 0
    lags: list[float] = []
    for s0, s1, gidx in episodes:
        ep_steps = range(int(s0), int(s1) + 1)
        baseline_vals = np.asarray([altru_lookup.get((st, int(gidx)), np.nan) for st in ep_steps], dtype=float)
        baseline = float(np.nanmean(baseline_vals))
        if not np.isfinite(baseline):
            baseline = 0.0
        fut_lo = int(s1 + 1)
        fut_hi = int(s1 + int(recovery_window_steps))
        if fut_lo > fut_hi:
            continue
        fsteps = list(range(fut_lo, fut_hi + 1))
        alt_series = np.asarray([altru_lookup.get((st, int(gidx)), np.nan) for st in fsteps], dtype=float)
        alt_ok_mask = np.isfinite(alt_series) & (alt_series >= float(min_altruism_share)) & (
            (alt_series - baseline) >= float(min_altruism_lift)
        )
        if not bool(np.any(alt_ok_mask)):
            continue
        n_surge += 1
        first_surge_step = int(fsteps[int(np.argmax(alt_ok_mask))])
        post = seq.loc[
            (seq["step"] >= int(first_surge_step))
            & (seq["step"] <= int(fut_hi))
            & (seq["conflict"] == True)
        ].copy()
        if len(post) == 0:
            continue
        post_margin = pd.to_numeric(post["margin"], errors="coerce").to_numpy(dtype=float)
        post_step = pd.to_numeric(post["step"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(post_margin) & np.isfinite(post_step) & (
            post_margin > float(margin_puzzle_recovery_threshold)
        )
        if not bool(np.any(mask)):
            continue
        rec_step = int(post_step[np.argmax(mask)])
        n_recovery += 1
        lags.append(float(rec_step - s1))

    out["lockin_episode_count_conflict"] = float(n_episode)
    out["lockin_altruism_surge_count_conflict"] = float(n_surge)
    out["lockin_recovery_event_count_conflict"] = float(n_recovery)
    out["lockin_altruism_surge_share_conflict"] = (
        float(n_surge) / float(n_episode) if n_episode > 0 else np.nan
    )
    out["lockin_recovery_share_conflict"] = (
        float(n_recovery) / float(n_episode) if n_episode > 0 else np.nan
    )
    out["lockin_recovery_lag_mean_steps_conflict"] = float(np.mean(lags)) if lags else np.nan
    return out


def _compute_moderate_recovery_metrics_for_run(
    *,
    run_dir: Path,
    agents: pd.DataFrame,
    min_group_size: int,
    takeover_loss_frac_min: float,
    takeover_capture_frac_min: float,
    rebound_loss_frac_min: float,
    rebound_share_vs_a_peak_min: float,
    min_steps: int,
    min_peak_share: float,
    length_bonus_max: float,
    altruism_bonus_weight: float,
    altruism_bonus_floor: float,
) -> dict[str, float]:
    out = {
        "moderate_recovery_event_count": 0.0,
        "moderate_recovery_pair_count": 0.0,
        "moderate_recovery_pair_coverage": 0.0,
        "moderate_recovery_strength_best": 0.0,
        "moderate_recovery_strength_top3_mean": 0.0,
        "moderate_recovery_strength_run": 0.0,
    }
    req = {"step", "agent_id", "personality_group_idx", "participating"}
    if not req <= set(agents.columns):
        return out

    ap = agents.copy()
    if "eligible_for_election" in ap.columns:
        ap = ap.loc[ap["eligible_for_election"] == True]
    if len(ap) == 0:
        return out

    group_sizes = (
        ap.loc[:, ["agent_id", "personality_group_idx"]]
        .drop_duplicates(subset=["agent_id"])
        .groupby("personality_group_idx")
        .size()
        .sort_index()
    )
    groups = [int(g) for g in group_sizes[group_sizes >= int(min_group_size)].index.tolist()]
    if len(groups) < 2:
        return out

    steps = np.arange(int(ap["step"].min()), int(ap["step"].max()) + 1, dtype=int)
    if steps.size < max(4, int(min_steps) + 2):
        return out
    step0 = int(steps[0])
    gidx = {int(g): i for i, g in enumerate(groups)}
    n_steps = int(steps.size)
    n_groups = int(len(groups))

    turnout_rel = np.zeros((n_steps, n_groups), dtype=np.float64)
    participants = (
        ap.loc[ap["participating"] == True]
        .groupby(["step", "personality_group_idx"])
        .size()
    )
    for (st, grp), n_part in participants.items():
        gi = gidx.get(int(grp))
        if gi is None:
            continue
        si = int(st) - step0
        if 0 <= si < n_steps:
            turnout_rel[si, gi] = float(n_part) / float(group_sizes[int(grp)])

    group_sizes_vec = np.asarray([float(group_sizes[int(g)]) for g in groups], dtype=np.float64)
    turnout_counts = turnout_rel * group_sizes_vec.reshape(1, -1)
    total_counts = turnout_counts.sum(axis=1, keepdims=True)
    safe_total = np.where(total_counts > 1e-12, total_counts, 1.0)
    turnout_share = turnout_counts / safe_total

    altru_lookup: dict[tuple[int, int], float] = {}
    abw = float(max(0.0, altruism_bonus_weight))
    abf = float(np.clip(altruism_bonus_floor, 0.0, 0.99))
    if abw > 0.0:
        votes_path = run_dir / "votes.parquet"
        if votes_path.exists():
            try:
                votes = pd.read_parquet(votes_path, columns=["step", "agent_id", "voted_altruistically"])
                amap = ap.loc[:, ["step", "agent_id", "personality_group_idx"]].drop_duplicates(
                    subset=["step", "agent_id"]
                )
                vm = votes.merge(amap, on=["step", "agent_id"], how="inner")
                if len(vm) > 0:
                    vm["voted_altruistically"] = pd.to_numeric(vm["voted_altruistically"], errors="coerce")
                    g = (
                        vm.groupby(["step", "personality_group_idx"], as_index=False)
                        .agg(group_altruism_share=("voted_altruistically", "mean"))
                    )
                    for row in g.itertuples(index=False):
                        st = int(getattr(row, "step"))
                        gi = int(getattr(row, "personality_group_idx"))
                        av = float(getattr(row, "group_altruism_share"))
                        if np.isfinite(av):
                            altru_lookup[(st, gi)] = av
            except Exception:
                altru_lookup = {}

    event_strengths: list[float] = []
    pair_hits: set[tuple[int, int]] = set()
    min_steps_i = int(max(1, min_steps))
    length_bonus_max_f = float(max(0.0, length_bonus_max))
    eps = 1e-12

    for a_j, a_group in enumerate(groups):
        for b_j, b_group in enumerate(groups):
            if a_j == b_j:
                continue
            b_peak_idx = int(np.argmax(turnout_share[:, b_j]))
            b_peak = float(turnout_share[b_peak_idx, b_j])
            if b_peak < float(min_peak_share):
                continue
            a_at_b_peak = float(turnout_share[b_peak_idx, a_j])

            after_b_peak = np.arange(b_peak_idx + 1, n_steps, dtype=int)
            if after_b_peak.size == 0:
                continue
            b_vals = turnout_share[after_b_peak, b_j]
            a_vals = turnout_share[after_b_peak, a_j]
            b_loss = b_peak - b_vals
            a_gain = a_vals - a_at_b_peak
            phase1_mask = (
                (b_loss >= float(takeover_loss_frac_min) * b_peak)
                & (a_gain >= float(takeover_capture_frac_min) * b_loss)
            )
            phase1_streak = _first_true_streak(phase1_mask, min_steps_i)
            if phase1_streak is None:
                continue
            s1, e1 = phase1_streak
            t1_start = int(after_b_peak[s1])
            t1_end = int(after_b_peak[e1])

            post1 = np.arange(t1_end + 1, n_steps, dtype=int)
            if post1.size == 0:
                continue
            a_post = turnout_share[post1, a_j]
            a_peak_rel_idx = int(np.argmax(a_post))
            t_a_peak = int(post1[a_peak_rel_idx])
            a_peak = float(turnout_share[t_a_peak, a_j])
            if a_peak < float(min_peak_share):
                continue

            after_a_peak = np.arange(t_a_peak + 1, n_steps, dtype=int)
            if after_a_peak.size == 0:
                continue
            a_after = turnout_share[after_a_peak, a_j]
            b_after = turnout_share[after_a_peak, b_j]
            phase2_mask = (
                ((a_peak - a_after) >= float(rebound_loss_frac_min) * a_peak)
                & (b_after >= float(rebound_share_vs_a_peak_min) * a_peak)
            )
            phase2_streak = _first_true_streak(phase2_mask, min_steps_i)
            if phase2_streak is None:
                continue
            s2, e2 = phase2_streak
            t2_start = int(after_a_peak[s2])
            t2_end = int(after_a_peak[e2])

            b_loss_frac = float((b_peak - turnout_share[t1_end, b_j]) / max(b_peak, eps))
            a_capture_frac = float(
                max(0.0, turnout_share[t1_end, a_j] - a_at_b_peak)
                / max(max(0.0, b_peak - turnout_share[t1_end, b_j]), eps)
            )
            a_loss_frac = float((a_peak - turnout_share[t2_end, a_j]) / max(a_peak, eps))
            b_rebound_vs_a_peak = float(turnout_share[t2_end, b_j] / max(a_peak, eps))

            phase1_strength = float(
                np.clip(0.60 * b_loss_frac + 0.40 * np.clip(a_capture_frac, 0.0, 1.0), 0.0, 1.0)
            )
            phase2_strength = float(
                np.clip(0.60 * a_loss_frac + 0.40 * np.clip(b_rebound_vs_a_peak, 0.0, 1.0), 0.0, 1.0)
            )
            peak_factor = float(np.clip(min(b_peak, a_peak) / 0.25, 0.0, 1.0))
            len1 = int(t1_end - t1_start + 1)
            len2 = int(t2_end - t2_start + 1)
            len_ratio = float(max(0.0, min(len1, len2) - min_steps_i) / float(max(min_steps_i, 1)))
            len_boost = float(1.0 + length_bonus_max_f * np.clip(len_ratio, 0.0, 1.0))
            altruism_boost = 1.0
            if abw > 0.0 and altru_lookup:
                a_vals = np.asarray(
                    [altru_lookup.get((int(step0 + st), int(b_group)), np.nan) for st in range(t2_start, t2_end + 1)],
                    dtype=float,
                )
                a_mean = float(np.nanmean(a_vals)) if np.isfinite(a_vals).any() else np.nan
                if np.isfinite(a_mean):
                    align = float(np.clip((a_mean - abf) / max(1e-12, 1.0 - abf), 0.0, 1.0))
                    altruism_boost = float(1.0 + abw * align)
            ev_strength = float(
                np.clip(np.sqrt(phase1_strength * phase2_strength) * peak_factor * len_boost * altruism_boost, 0.0, 1.0)
            )
            if ev_strength <= 0.0:
                continue
            event_strengths.append(ev_strength)
            pair_hits.add((int(a_group), int(b_group)))

    if not event_strengths:
        return out
    event_arr = np.asarray(sorted(event_strengths, reverse=True), dtype=float)
    topk = event_arr[: min(3, event_arr.size)]
    best = float(event_arr[0])
    top3 = float(np.mean(topk))
    denom_pairs = float(n_groups * max(n_groups - 1, 1))
    pair_cov = float(len(pair_hits)) / denom_pairs if denom_pairs > 0.0 else 0.0
    run_strength = float(np.clip((0.65 * best + 0.35 * top3) * (0.6 + 0.4 * np.sqrt(pair_cov)), 0.0, 1.0))

    out["moderate_recovery_event_count"] = float(event_arr.size)
    out["moderate_recovery_pair_count"] = float(len(pair_hits))
    out["moderate_recovery_pair_coverage"] = float(pair_cov)
    out["moderate_recovery_strength_best"] = float(best)
    out["moderate_recovery_strength_top3_mean"] = float(top3)
    out["moderate_recovery_strength_run"] = float(run_strength)
    return out


def compute_run_features_from_tables(
    area_steps: pd.DataFrame,
    agents: pd.DataFrame,
    *,
    burn_in_steps: int = 0,
) -> dict[str, float]:
    req_area = {"step", "participants", "turnout", "gini_index", "dist_to_reality", "winning_option_id"}
    req_agents = {"step", "personality_group_idx", "participating"}
    missing_area = sorted(req_area - set(area_steps.columns))
    missing_agents = sorted(req_agents - set(agents.columns))
    if missing_area:
        raise ValueError(f"area_steps missing columns: {missing_area}")
    if missing_agents:
        raise ValueError(f"agents missing columns: {missing_agents}")

    quality_col = "quality_distance"
    if "quality_distance" not in area_steps.columns:
        quality_col = "puzzle_distance" if "puzzle_distance" in area_steps.columns else "dist_to_reality"
        area_steps = area_steps.copy()
        area_steps["quality_distance"] = pd.to_numeric(area_steps[quality_col], errors="coerce")
        if quality_col == "puzzle_distance" and not np.isfinite(
            area_steps["quality_distance"].to_numpy(dtype=float)
        ).any():
            area_steps["quality_distance"] = pd.to_numeric(
                area_steps["dist_to_reality"], errors="coerce"
            )

    per_step = (
        area_steps.groupby("step", as_index=False)
        .agg(
            participants=("participants", "sum"),
            turnout=("turnout", "mean"),
            gini_index=("gini_index", "mean"),
            dist_to_reality=("dist_to_reality", "mean"),
            quality_distance=("quality_distance", "mean"),
            winning_option_id=("winning_option_id", "first"),
        )
        .sort_values("step")
        .reset_index(drop=True)
    )
    if len(per_step) == 0:
        raise ValueError("area_steps has no rows.")

    turnout_series = pd.to_numeric(per_step["turnout"], errors="coerce").astype(float)
    n_steps = int(len(turnout_series))
    # Step-count independent windows: use a fractional window with a small floor, capped by run length.
    win = max(1, min(n_steps, max(5, int(np.ceil(0.10 * n_steps)))))
    turnout_start_window_mean = float(turnout_series.iloc[:win].mean())
    turnout_end_window_mean = float(turnout_series.iloc[-win:].mean())
    turnout_drop_start_end = float(turnout_start_window_mean - turnout_end_window_mean)
    late_steps = set(pd.to_numeric(per_step["step"].iloc[-win:], errors="coerce").astype(int).tolist())
    turnout_outside_20_80_share = float(
        np.mean((turnout_series.to_numpy(dtype=float) < 20.0) | (turnout_series.to_numpy(dtype=float) > 80.0))
    )
    turnout_trend_slope_norm = 0.0
    turnout_decline_slope_norm = 0.0
    if n_steps >= 2:
        x = np.linspace(0.0, 1.0, num=n_steps, dtype=float)
        y = (turnout_series.to_numpy(dtype=float) / 100.0).astype(float)
        if np.isfinite(y).all():
            coeffs = np.polyfit(x, y, 1)
            turnout_trend_slope_norm = float(coeffs[0])
            turnout_decline_slope_norm = float(max(0.0, -turnout_trend_slope_norm))

    abstain_mask = (per_step["participants"].to_numpy(dtype=float) <= 0.0)
    max_all_abstain_stretch = _max_true_stretch(abstain_mask)

    post = per_step.loc[per_step["step"] > int(burn_in_steps), "winning_option_id"].to_numpy()
    winner_changes_post_burnin = _winner_changes(post)
    post_transition_count = max(0, int(len(post)) - 1)
    winner_change_rate_post_burnin = (
        float(winner_changes_post_burnin) / float(post_transition_count)
        if post_transition_count > 0
        else 0.0
    )
    winner_entropy_norm = _winner_entropy_norm(post if len(post) > 0 else per_step["winning_option_id"].to_numpy())
    dist_vals = pd.to_numeric(per_step["quality_distance"], errors="coerce").to_numpy(dtype=float)
    finite_dist = np.isfinite(dist_vals)
    if finite_dist.any():
        dist_nonzero_share = float(np.mean(dist_vals[finite_dist] > 1e-12))
    else:
        dist_nonzero_share = 0.0

    group_participation = agents.groupby("personality_group_idx", as_index=False).agg(
        p=("participating", "mean")
    )["p"]
    group_std = _safe_std(group_participation)
    group_range = float(group_participation.max() - group_participation.min()) if len(group_participation) else 0.0

    # Per-step group turnout (resident denominator via full group rows per step)
    group_step = (
        agents.groupby(["step", "personality_group_idx"], as_index=False)
        .agg(turnout_resident=("participating", "mean"))
        .sort_values(["step", "personality_group_idx"])
    )
    if len(group_step) > 0:
        step_range = group_step.groupby("step", as_index=False).agg(
            group_turnout_range=("turnout_resident", lambda s: float(np.nanmax(s) - np.nanmin(s)))
        )
        group_turnout_range_mean = float(step_range["group_turnout_range"].mean())
    else:
        group_turnout_range_mean = 0.0

    global_step = agents.groupby("step", as_index=False).agg(global_turnout=("participating", "mean"))
    merged_step = group_step.merge(global_step, on="step", how="left")
    if len(merged_step) > 0:
        merged_step["resid_abs"] = (merged_step["turnout_resident"] - merged_step["global_turnout"]).abs()
        group_turnout_residual_abs_mean = float(merged_step["resid_abs"].mean())
    else:
        group_turnout_residual_abs_mean = 0.0

    roll3_group_turnout_range_mean = 0.0
    roll3_group_turnout_range_max = 0.0
    roll20_group_turnout_range_mean = 0.0
    roll20_group_turnout_range_max = 0.0
    competitive_step_share = 0.0
    piv = group_step.pivot(index="step", columns="personality_group_idx", values="turnout_resident").sort_index()
    arr = piv.to_numpy(dtype=float)
    if arr.shape[0] >= 20 and arr.shape[1] >= 2:
        roll_cols = [
            np.convolve(arr[:, i], np.ones(20, dtype=float) / 20.0, mode="valid")
            for i in range(arr.shape[1])
        ]
        roll = np.vstack(roll_cols).T
        r = np.nanmax(roll, axis=1) - np.nanmin(roll, axis=1)
        if np.isfinite(r).any():
            roll20_group_turnout_range_mean = float(np.nanmean(r))
            roll20_group_turnout_range_max = float(np.nanmax(r))
    if arr.shape[0] >= 3 and arr.shape[1] >= 2:
        roll_cols = [
            np.convolve(arr[:, i], np.ones(3, dtype=float) / 3.0, mode="valid")
            for i in range(arr.shape[1])
        ]
        roll = np.vstack(roll_cols).T
        r = np.nanmax(roll, axis=1) - np.nanmin(roll, axis=1)
        if np.isfinite(r).any():
            roll3_group_turnout_range_mean = float(np.nanmean(r))
            roll3_group_turnout_range_max = float(np.nanmax(r))
    if arr.shape[0] >= 2 and arr.shape[1] >= 2:
        d = np.diff(arr, axis=0)
        valid = np.all(np.isfinite(d), axis=1)
        if valid.any():
            up = np.any(d[valid] > 1e-12, axis=1)
            down = np.any(d[valid] < -1e-12, axis=1)
            competitive_step_share = float(np.mean(up & down))

    roll10_dist_std_mean = 0.0
    if len(per_step) >= 10:
        v = per_step["quality_distance"].rolling(window=10, min_periods=10).std()
        v = pd.to_numeric(v, errors="coerce")
        if v.notna().any():
            roll10_dist_std_mean = float(v.mean(skipna=True))

    roll10_winner_change_rate = 0.0
    if len(per_step) >= 10:
        wins = per_step["winning_option_id"].to_numpy()
        rates: list[float] = []
        for i in range(0, len(wins) - 9):
            w = wins[i : i + 10]
            rates.append(float(np.mean(w[1:] != w[:-1])))
        if rates:
            roll10_winner_change_rate = float(np.mean(rates))

    roll10_turnout_slope_abs_mean = 0.0
    if len(per_step) >= 10:
        t = per_step["step"].to_numpy(dtype=float)
        y = per_step["turnout"].to_numpy(dtype=float)
        slopes: list[float] = []
        for i in range(0, len(per_step) - 9):
            tw = t[i : i + 10]
            yw = y[i : i + 10]
            if np.all(np.isfinite(tw)) and np.all(np.isfinite(yw)):
                coeffs = np.polyfit(tw, yw, 1)
                slopes.append(float(abs(coeffs[0])))
        if slopes:
            roll10_turnout_slope_abs_mean = float(np.mean(slopes))

    roll10_group_sync_index = 0.0
    if arr.shape[0] >= 10 and arr.shape[1] >= 2:
        sync_vals: list[float] = []
        for i in range(0, arr.shape[0] - 9):
            w = arr[i : i + 10, :]
            mean_corr = _mean_finite_pairwise_corr(w)
            if mean_corr is not None:
                sync_vals.append(mean_corr)
        if sync_vals:
            roll10_group_sync_index = float(np.mean(sync_vals))

    # Participant-composition dynamics (participants split across groups each step).
    # This captures temporal reallocation between groups, not just cross-sectional spread.
    participant_share_max_abs_drift_20 = 0.0
    participant_share_mean_abs_drift_20_w = 0.0
    participant_share_turnover_rate_w = 0.0
    if len(group_step) > 0:
        group_part = (
            agents.groupby(["step", "personality_group_idx"], as_index=False)
            .agg(participants=("participating", "sum"))
            .sort_values(["step", "personality_group_idx"])
        )
        if len(group_part) > 0:
            totals = group_part.groupby("step", as_index=False).agg(total=("participants", "sum"))
            group_part = group_part.merge(totals, on="step", how="left")
            group_part["participant_share"] = np.where(
                pd.to_numeric(group_part["total"], errors="coerce").to_numpy(dtype=float) > 0.0,
                pd.to_numeric(group_part["participants"], errors="coerce").to_numpy(dtype=float)
                / pd.to_numeric(group_part["total"], errors="coerce").to_numpy(dtype=float),
                np.nan,
            )
            comp_piv = (
                group_part.pivot(index="step", columns="personality_group_idx", values="participant_share")
                .sort_index()
            )
            if len(comp_piv) > 0 and comp_piv.shape[1] > 0:
                comp_arr = comp_piv.to_numpy(dtype=float)
                n_comp_steps = int(comp_arr.shape[0])
                win_comp = max(1, min(n_comp_steps, max(5, int(np.ceil(0.20 * n_comp_steps)))))

                # Size-weights by group prevalence in agent table (stable proxy for group size).
                g_weights = (
                    agents.groupby("personality_group_idx", as_index=False)
                    .agg(n=("participating", "count"))
                    .set_index("personality_group_idx")["n"]
                )
                weight_vec = np.array(
                    [
                        float(g_weights.get(int(g), 0.0))
                        for g in comp_piv.columns.to_numpy(dtype=int)
                    ],
                    dtype=float,
                )
                if np.isfinite(weight_vec).any() and float(np.nansum(weight_vec)) > 0.0:
                    weight_vec = np.where(np.isfinite(weight_vec), weight_vec, 0.0)
                    weight_vec = weight_vec / float(np.sum(weight_vec))
                else:
                    weight_vec = np.full(comp_arr.shape[1], 1.0 / float(comp_arr.shape[1]), dtype=float)

                drift_vals: list[float] = []
                turnover_vals: list[float] = []
                for gi in range(comp_arr.shape[1]):
                    s = comp_arr[:, gi]
                    finite = np.isfinite(s)
                    if finite.sum() <= 1:
                        continue
                    sf = s[finite]
                    w = min(win_comp, int(sf.size))
                    early = float(np.nanmean(sf[:w]))
                    late = float(np.nanmean(sf[-w:]))
                    drift_vals.append(abs(late - early))

                    dif = np.abs(np.diff(sf))
                    if dif.size > 0:
                        turnover_vals.append(float(np.nanmean(dif)))
                    else:
                        turnover_vals.append(0.0)

                if drift_vals:
                    participant_share_max_abs_drift_20 = float(np.nanmax(drift_vals))
                    # Align weights to valid groups only.
                    valid_idx = [
                        gi for gi in range(comp_arr.shape[1])
                        if np.isfinite(comp_arr[:, gi]).sum() > 1
                    ]
                    if valid_idx:
                        wv = weight_vec[np.asarray(valid_idx, dtype=int)]
                        if float(np.sum(wv)) > 0.0:
                            wv = wv / float(np.sum(wv))
                            d = np.asarray(drift_vals, dtype=float)
                            participant_share_mean_abs_drift_20_w = float(np.sum(wv * d))
                if turnover_vals:
                    valid_idx = [
                        gi for gi in range(comp_arr.shape[1])
                        if np.isfinite(comp_arr[:, gi]).sum() > 1
                    ]
                    if valid_idx:
                        wv = weight_vec[np.asarray(valid_idx, dtype=int)]
                        if float(np.sum(wv)) > 0.0:
                            wv = wv / float(np.sum(wv))
                            t = np.asarray(turnover_vals, dtype=float)
                            participant_share_turnover_rate_w = float(np.sum(wv * t))

    participant_mask = agents["participating"].astype(bool)
    abstainer_mask = ~participant_mask

    def _gap_abs(col: str) -> float:
        if col not in agents.columns:
            return 0.0
        a = agents.loc[participant_mask, col]
        b = agents.loc[abstainer_mask, col]
        if len(a) == 0 or len(b) == 0:
            return 0.0
        av = float(pd.to_numeric(a, errors="coerce").mean())
        bv = float(pd.to_numeric(b, errors="coerce").mean())
        if not np.isfinite(av) or not np.isfinite(bv):
            return 0.0
        return float(abs(av - bv))

    # Group-level participant-vs-abstainer relative payoff gap.
    group_pa_delta_rel_gap_abs = 0.0
    if "election_delta_rel" in agents.columns:
        pool = agents.copy()
        if "eligible_for_election" in pool.columns:
            pool = pool.loc[pool["eligible_for_election"] == True]
        if len(pool) > 0:
            grp = (
                pool.groupby(["step", "personality_group_idx", "participating"], as_index=False)
                .agg(m=("election_delta_rel", "mean"))
            )
            piv = grp.pivot_table(
                index=["step", "personality_group_idx"],
                columns="participating",
                values="m",
                aggfunc="mean",
            )
            if (True in piv.columns) and (False in piv.columns):
                gaps = (piv[True] - piv[False]).abs()
                if gaps.notna().any():
                    group_pa_delta_rel_gap_abs = float(gaps.mean())

    def _lag1_group_signal_turnout_corr(signal_col: str) -> float:
        if (signal_col not in agents.columns) or (len(group_step) == 0):
            return 0.0
        sig = (
            agents.groupby(["step", "personality_group_idx"], as_index=False)
            .agg(signal=(signal_col, "mean"))
            .sort_values(["personality_group_idx", "step"])
        )
        tur = group_step.rename(columns={"turnout_resident": "turnout"}).sort_values(
            ["personality_group_idx", "step"]
        )
        merged = sig.merge(tur, on=["step", "personality_group_idx"], how="inner").sort_values(
            ["personality_group_idx", "step"]
        )
        if len(merged) == 0:
            return 0.0
        merged["turnout_next"] = merged.groupby("personality_group_idx")["turnout"].shift(-1)
        merged["turnout_delta_next"] = merged["turnout_next"] - merged["turnout"]
        valid = merged[["signal", "turnout_delta_next"]].dropna()
        if len(valid) < 3:
            return 0.0
        x = valid["signal"].to_numpy(dtype=float)
        y = valid["turnout_delta_next"].to_numpy(dtype=float)
        if np.std(x) <= 0.0 or np.std(y) <= 0.0:
            return 0.0
        c = float(np.corrcoef(x, y)[0, 1])
        return float(c) if np.isfinite(c) else 0.0

    lag1_group_signal_turnout_response_corr = _lag1_group_signal_turnout_corr("election_delta_rel")
    lag1_participation_signal_turnout_response_corr = _lag1_group_signal_turnout_corr("participation_signal")
    lag1_participation_signal_group_component_turnout_response_corr = _lag1_group_signal_turnout_corr(
        "participation_signal_group_component"
    )
    lag1_participation_signal_fee_component_turnout_response_corr = _lag1_group_signal_turnout_corr(
        "participation_signal_fee_component"
    )

    participation_q_delta_mean = np.nan
    participation_q_delta_mean_abs = np.nan
    participation_q_delta_late_window_mean = np.nan
    participation_q_delta_late_window_mean_abs = np.nan
    participation_q_delta_group_dispersion_late_w = np.nan
    if {"agent_id", "q_participation", "step", "personality_group_idx"} <= set(agents.columns):
        qpool = agents.copy()
        if "eligible_for_election" in qpool.columns:
            qpool = qpool.loc[qpool["eligible_for_election"] == True]
        qpool = qpool.loc[:, ["agent_id", "step", "personality_group_idx", "q_participation"]].copy()
        qpool["q_participation"] = pd.to_numeric(qpool["q_participation"], errors="coerce")
        qpool = qpool.dropna(subset=["q_participation"]).sort_values(["agent_id", "step"])
        if len(qpool) > 0:
            qpool["participation_q_delta_exact"] = (
                qpool.groupby("agent_id", sort=False)["q_participation"].diff()
            )
            dq = pd.to_numeric(qpool["participation_q_delta_exact"], errors="coerce")
            valid = qpool.loc[dq.notna(), ["step", "personality_group_idx", "participation_q_delta_exact"]].copy()
            if len(valid) > 0:
                v = pd.to_numeric(valid["participation_q_delta_exact"], errors="coerce").to_numpy(dtype=float)
                participation_q_delta_mean = float(np.nanmean(v))
                participation_q_delta_mean_abs = float(np.nanmean(np.abs(v)))
                late_mask = valid["step"].astype(int).isin(late_steps).to_numpy()
                if late_mask.any():
                    lv = v[late_mask]
                    if lv.size > 0:
                        participation_q_delta_late_window_mean = float(np.nanmean(lv))
                        participation_q_delta_late_window_mean_abs = float(np.nanmean(np.abs(lv)))
                    gl = valid.loc[late_mask].groupby(
                        ["step", "personality_group_idx"], as_index=False
                    ).agg(
                        q_mean=("participation_q_delta_exact", "mean"),
                        n=("participation_q_delta_exact", "count"),
                    )
                    if len(gl) > 0:
                        disp_vals: list[float] = []
                        for _, gstep in gl.groupby("step", sort=False):
                            if len(gstep) < 2:
                                continue
                            x = pd.to_numeric(gstep["q_mean"], errors="coerce").to_numpy(dtype=float)
                            w = pd.to_numeric(gstep["n"], errors="coerce").to_numpy(dtype=float)
                            mask = np.isfinite(x) & np.isfinite(w) & (w > 0.0)
                            if mask.sum() < 2:
                                continue
                            x = x[mask]
                            w = w[mask]
                            wsum = float(w.sum())
                            if wsum <= 0.0:
                                continue
                            mu = float(np.sum(w * x) / wsum)
                            var = float(np.sum(w * (x - mu) ** 2) / wsum)
                            disp_vals.append(float(np.sqrt(max(0.0, var))))
                        if disp_vals:
                            participation_q_delta_group_dispersion_late_w = float(np.mean(disp_vals))

    return {
        "mean_turnout": float(per_step["turnout"].mean()),
        "turnout_std": _safe_std(per_step["turnout"]),
        "turnout_start_window_mean": float(turnout_start_window_mean),
        "turnout_end_window_mean": float(turnout_end_window_mean),
        "turnout_drop_start_end": float(turnout_drop_start_end),
        "turnout_outside_20_80_share": float(turnout_outside_20_80_share),
        "turnout_trend_slope_norm": float(turnout_trend_slope_norm),
        "turnout_decline_slope_norm": float(turnout_decline_slope_norm),
        "mean_gini": float(per_step["gini_index"].mean()),
        "gini_std": _safe_std(per_step["gini_index"]),
        "mean_dist": float(per_step["quality_distance"].mean()),
        "dist_std": _safe_std(per_step["quality_distance"]),
        "max_all_abstain_stretch": float(max_all_abstain_stretch),
        "winner_changes_post_burnin": float(winner_changes_post_burnin),
        "winner_change_rate_post_burnin": float(winner_change_rate_post_burnin),
        "winner_entropy_norm": float(winner_entropy_norm),
        "dist_nonzero_share": float(dist_nonzero_share),
        "competitive_step_share": float(competitive_step_share),
        "group_participation_std": float(group_std),
        "group_participation_range": float(group_range),
        "group_turnout_range_mean": float(group_turnout_range_mean),
        "roll3_group_turnout_range_mean": float(roll3_group_turnout_range_mean),
        "roll3_group_turnout_range_max": float(roll3_group_turnout_range_max),
        "roll20_group_turnout_range_mean": float(roll20_group_turnout_range_mean),
        "roll20_group_turnout_range_max": float(roll20_group_turnout_range_max),
        "roll10_dist_std_mean": float(roll10_dist_std_mean),
        "roll10_winner_change_rate": float(roll10_winner_change_rate),
        "roll10_turnout_slope_abs_mean": float(roll10_turnout_slope_abs_mean),
        "roll10_group_sync_index": float(roll10_group_sync_index),
        "lag1_group_signal_turnout_response_corr": float(lag1_group_signal_turnout_response_corr),
        "lag1_participation_signal_turnout_response_corr": float(
            lag1_participation_signal_turnout_response_corr
        ),
        "lag1_participation_signal_group_component_turnout_response_corr": float(
            lag1_participation_signal_group_component_turnout_response_corr
        ),
        "lag1_participation_signal_fee_component_turnout_response_corr": float(
            lag1_participation_signal_fee_component_turnout_response_corr
        ),
        "participant_share_max_abs_drift_20": float(participant_share_max_abs_drift_20),
        "participant_share_mean_abs_drift_20_w": float(participant_share_mean_abs_drift_20_w),
        "participant_share_turnover_rate_w": float(participant_share_turnover_rate_w),
        "participation_q_delta_mean": float(participation_q_delta_mean)
        if np.isfinite(participation_q_delta_mean)
        else float("nan"),
        "participation_q_delta_mean_abs": float(participation_q_delta_mean_abs)
        if np.isfinite(participation_q_delta_mean_abs)
        else float("nan"),
        "participation_q_delta_late_window_mean": float(participation_q_delta_late_window_mean)
        if np.isfinite(participation_q_delta_late_window_mean)
        else float("nan"),
        "participation_q_delta_late_window_mean_abs": float(participation_q_delta_late_window_mean_abs)
        if np.isfinite(participation_q_delta_late_window_mean_abs)
        else float("nan"),
        "participation_q_delta_group_dispersion_late_w": float(
            participation_q_delta_group_dispersion_late_w
        )
        if np.isfinite(participation_q_delta_group_dispersion_late_w)
        else float("nan"),
        "group_turnout_residual_abs_mean": float(group_turnout_residual_abs_mean),
        "participant_abstainer_delta_rel_gap_abs": _gap_abs("election_delta_rel"),
        "group_participant_abstainer_delta_rel_gap_abs": float(group_pa_delta_rel_gap_abs),
        "participant_abstainer_delta_abs_gap_abs": _gap_abs("election_delta_abs"),
        "participant_abstainer_diss_signal_gap_abs": _gap_abs("dissatisfaction_signal"),
    }


def collect_run_features(
    doe_root: Path,
    *,
    burn_in_steps: int = 0,
    puzzle_conflict_min_dist: float = DEFAULT_SCORING_THRESHOLDS["puzzle_conflict_min_dist"],
    lockin_episode_min_len_steps: int = int(DEFAULT_SCORING_THRESHOLDS["lockin_episode_min_len_steps"]),
    lockin_recovery_window_steps: int = int(DEFAULT_SCORING_THRESHOLDS["lockin_recovery_window_steps"]),
    lockin_min_dominant_participant_share: float = DEFAULT_SCORING_THRESHOLDS["lockin_min_dominant_participant_share"],
    lockin_min_altruism_share: float = DEFAULT_SCORING_THRESHOLDS["lockin_min_altruism_share"],
    lockin_min_altruism_lift: float = DEFAULT_SCORING_THRESHOLDS["lockin_min_altruism_lift"],
    lockin_margin_power_threshold: float = DEFAULT_SCORING_THRESHOLDS["lockin_margin_power_threshold"],
    lockin_margin_puzzle_recovery_threshold: float = DEFAULT_SCORING_THRESHOLDS["lockin_margin_puzzle_recovery_threshold"],
    moderate_min_group_size: int = int(DEFAULT_SCORING_THRESHOLDS["moderate_min_group_size"]),
    moderate_takeover_loss_frac_min: float = DEFAULT_SCORING_THRESHOLDS["moderate_takeover_loss_frac_min"],
    moderate_takeover_capture_frac_min: float = DEFAULT_SCORING_THRESHOLDS["moderate_takeover_capture_frac_min"],
    moderate_rebound_loss_frac_min: float = DEFAULT_SCORING_THRESHOLDS["moderate_rebound_loss_frac_min"],
    moderate_rebound_share_vs_a_peak_min: float = DEFAULT_SCORING_THRESHOLDS["moderate_rebound_share_vs_a_peak_min"],
    moderate_min_steps: int = int(DEFAULT_SCORING_THRESHOLDS["moderate_min_steps"]),
    moderate_min_peak_share: float = DEFAULT_SCORING_THRESHOLDS["moderate_min_peak_share"],
    moderate_length_bonus_max: float = DEFAULT_SCORING_THRESHOLDS["moderate_length_bonus_max"],
    moderate_altruism_bonus_weight: float = DEFAULT_SCORING_THRESHOLDS["moderate_altruism_bonus_weight"],
    moderate_altruism_bonus_floor: float = DEFAULT_SCORING_THRESHOLDS["moderate_altruism_bonus_floor"],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(doe_root.glob("design_*/rule_*/seed_*/run_0")):
        area_path = run_dir / "area_steps.parquet"
        agents_path = run_dir / "agents.parquet"
        if not area_path.exists() or not agents_path.exists():
            continue
        area = pd.read_parquet(area_path)
        agents = pd.read_parquet(agents_path)
        features = compute_run_features_from_tables(area, agents, burn_in_steps=burn_in_steps)
        features.update(
            _compute_puzzle_power_metrics_for_run(
                run_dir=run_dir,
                area_steps=area,
                agents=agents,
                burn_in_steps=int(burn_in_steps),
                conflict_min_dist=float(puzzle_conflict_min_dist),
            )
        )
        features.update(
            _compute_lockin_recovery_metrics_for_run(
                run_dir=run_dir,
                area_steps=area,
                agents=agents,
                burn_in_steps=int(burn_in_steps),
                conflict_min_dist=float(puzzle_conflict_min_dist),
                episode_min_len_steps=int(lockin_episode_min_len_steps),
                recovery_window_steps=int(lockin_recovery_window_steps),
                min_dominant_participant_share=float(lockin_min_dominant_participant_share),
                min_altruism_share=float(lockin_min_altruism_share),
                min_altruism_lift=float(lockin_min_altruism_lift),
                margin_power_threshold=float(lockin_margin_power_threshold),
                margin_puzzle_recovery_threshold=float(lockin_margin_puzzle_recovery_threshold),
            )
        )
        features.update(
            _compute_moderate_recovery_metrics_for_run(
                run_dir=run_dir,
                agents=agents,
                min_group_size=int(moderate_min_group_size),
                takeover_loss_frac_min=float(moderate_takeover_loss_frac_min),
                takeover_capture_frac_min=float(moderate_takeover_capture_frac_min),
                rebound_loss_frac_min=float(moderate_rebound_loss_frac_min),
                rebound_share_vs_a_peak_min=float(moderate_rebound_share_vs_a_peak_min),
                min_steps=int(moderate_min_steps),
                min_peak_share=float(moderate_min_peak_share),
                length_bonus_max=float(moderate_length_bonus_max),
                altruism_bonus_weight=float(moderate_altruism_bonus_weight),
                altruism_bonus_floor=float(moderate_altruism_bonus_floor),
            )
        )

        design_name = run_dir.parts[-4]
        rule_name = run_dir.parts[-3].replace("rule_", "", 1)
        seed_name = run_dir.parts[-2]
        rows.append(
            {
                "design_id": int(design_name.split("_")[1]),
                "rule_name": str(rule_name),
                "seed": int(seed_name.split("_")[1]),
                "run_dir": str(run_dir),
                **features,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No DOE run directories with area_steps/agents found under: {doe_root}")
    return pd.DataFrame(rows).sort_values(["design_id", "rule_name", "seed"]).reset_index(drop=True)


def apply_hard_gates(
    run_features: pd.DataFrame,
    *,
    max_all_abstain_stretch: float = DEFAULT_SCORING_THRESHOLDS["max_all_abstain_stretch"],
    min_winner_changes_post_burnin: float = DEFAULT_SCORING_THRESHOLDS["min_winner_changes_post_burnin"],
    max_winner_changes_post_burnin: float | None = None,  # legacy ignored (rate-based gate now)
    min_winner_change_rate_post_burnin: float = DEFAULT_SCORING_THRESHOLDS["min_winner_change_rate_post_burnin"],
    max_winner_change_rate_post_burnin: float = DEFAULT_SCORING_THRESHOLDS["max_winner_change_rate_post_burnin"],
    min_group_turnout_range_mean: float = DEFAULT_SCORING_THRESHOLDS["min_group_turnout_range_mean"],
    min_roll3_group_turnout_range_max: float = DEFAULT_SCORING_THRESHOLDS["min_roll3_group_turnout_range_max"],
    min_roll20_group_turnout_range_max: float = DEFAULT_SCORING_THRESHOLDS["min_roll20_group_turnout_range_max"],
    min_turnout_std: float = DEFAULT_SCORING_THRESHOLDS["min_turnout_std"],
    min_gini_std: float = DEFAULT_SCORING_THRESHOLDS["min_gini_std"],
    min_dist_std: float = DEFAULT_SCORING_THRESHOLDS["min_dist_std"],
    min_winner_entropy_norm: float = DEFAULT_SCORING_THRESHOLDS["min_winner_entropy_norm"],
    min_dist_nonzero_share: float = DEFAULT_SCORING_THRESHOLDS["min_dist_nonzero_share"],
    min_competitive_step_share: float = DEFAULT_SCORING_THRESHOLDS["min_competitive_step_share"],
    min_mean_turnout: float = DEFAULT_SCORING_THRESHOLDS["min_mean_turnout"],
    max_mean_turnout: float = DEFAULT_SCORING_THRESHOLDS["max_mean_turnout"],
    min_puzzle_conflict_step_share_for_gate: float = DEFAULT_SCORING_THRESHOLDS["min_puzzle_conflict_step_share_for_gate"],
    max_puzzle_dominance_share_conflict: float = DEFAULT_SCORING_THRESHOLDS["max_puzzle_dominance_share_conflict"],
    min_power_recovery_share_conflict: float = DEFAULT_SCORING_THRESHOLDS["min_power_recovery_share_conflict"],
) -> pd.DataFrame:
    df = run_features.copy()
    _require_columns(df, REQUIRED_HARD_GATE_COLUMNS, context="apply_hard_gates(run_features)")
    _require_numeric_notna(
        df,
        REQUIRED_HARD_GATE_COLUMNS,
        context="apply_hard_gates(run_features)",
    )
    for col in [
        "puzzle_conflict_step_share",
        "puzzle_dominance_share_conflict",
        "power_recovery_share_conflict",
        "puzzle_power_margin_mean_conflict",
    ]:
        if col not in df.columns:
            df[col] = np.nan
    df["gate_no_collapse"] = df["max_all_abstain_stretch"] <= float(max_all_abstain_stretch)
    df["gate_no_lockin"] = df["winner_changes_post_burnin"] >= float(min_winner_changes_post_burnin)
    df["gate_not_too_chaotic"] = (
        (df["winner_change_rate_post_burnin"] >= float(min_winner_change_rate_post_burnin))
        & (df["winner_change_rate_post_burnin"] <= float(max_winner_change_rate_post_burnin))
    )
    df["gate_group_divergence"] = df["group_turnout_range_mean"] >= float(min_group_turnout_range_mean)
    df["gate_roll3_divergence"] = (
        df["roll3_group_turnout_range_max"] >= float(min_roll3_group_turnout_range_max)
    )
    df["gate_roll20_divergence"] = (
        df["roll20_group_turnout_range_max"] >= float(min_roll20_group_turnout_range_max)
    )
    df["gate_winner_entropy"] = df["winner_entropy_norm"] >= float(min_winner_entropy_norm)
    df["gate_dist_activity"] = df["dist_nonzero_share"] >= float(min_dist_nonzero_share)
    df["gate_competitive_steps"] = df["competitive_step_share"] >= float(min_competitive_step_share)
    df["gate_turnout_band"] = (
        (df["mean_turnout"] >= float(min_mean_turnout))
        & (df["mean_turnout"] <= float(max_mean_turnout))
    )
    df["gate_signal_present"] = (
        (df["turnout_std"] >= float(min_turnout_std))
        | (df["gini_std"] >= float(min_gini_std))
        | (df["dist_std"] >= float(min_dist_std))
    )
    puzzle_metric_available = (
        df["puzzle_conflict_step_share"].notna()
        & df["puzzle_dominance_share_conflict"].notna()
        & df["power_recovery_share_conflict"].notna()
    )
    enough_conflict = df["puzzle_conflict_step_share"] >= float(min_puzzle_conflict_step_share_for_gate)
    anti_monopoly_ok = (
        (df["puzzle_dominance_share_conflict"] <= float(max_puzzle_dominance_share_conflict))
        & (df["power_recovery_share_conflict"] >= float(min_power_recovery_share_conflict))
    )
    # Mild puzzle anti-monopoly gate: only enforced for puzzle-logged runs with enough
    # conflict between puzzle and static power direction.
    df["gate_puzzle_anti_monopoly"] = np.where(
        (puzzle_metric_available & enough_conflict),
        anti_monopoly_ok,
        True,
    )
    df["passes_hard_gates"] = (
        df["gate_no_collapse"]
        & df["gate_no_lockin"]
        & df["gate_not_too_chaotic"]
        & df["gate_roll3_divergence"]
        & df["gate_roll20_divergence"]
        & df["gate_winner_entropy"]
        & df["gate_dist_activity"]
        & df["gate_competitive_steps"]
        & df["gate_turnout_band"]
        & df["gate_signal_present"]
        & df["gate_puzzle_anti_monopoly"]
    )
    return df


def score_designs(
    run_features: pd.DataFrame,
    *,
    primary_rule_name: str = "approval",
    robust_rule_name: str = "utilitarian",
    weights: dict[str, float] | None = None,
    stage_weights: dict[str, float] | None = None,
    required_primary_runs: int | None = None,
    required_matched_seed_pairs: int | None = None,
    min_winner_entropy_norm: float = DEFAULT_SCORING_THRESHOLDS["min_winner_entropy_norm"],
    min_competitive_step_share: float = DEFAULT_SCORING_THRESHOLDS["min_competitive_step_share"],
    puzzle_dominance_share_score_low: float = DEFAULT_SCORING_THRESHOLDS["puzzle_dominance_share_score_low"],
    puzzle_dominance_share_score_high: float = DEFAULT_SCORING_THRESHOLDS["puzzle_dominance_share_score_high"],
    moderate_selector_quality_gate_zero_at: float = DEFAULT_SCORING_THRESHOLDS["moderate_selector_quality_gate_zero_at"],
    moderate_selector_quality_gate_good_min: float = DEFAULT_SCORING_THRESHOLDS["moderate_selector_quality_gate_good_min"],
    turnout_start_score_low: float = DEFAULT_SCORING_THRESHOLDS["turnout_start_score_low"],
    turnout_start_score_high: float = DEFAULT_SCORING_THRESHOLDS["turnout_start_score_high"],
    turnout_end_score_low: float = DEFAULT_SCORING_THRESHOLDS["turnout_end_score_low"],
    turnout_end_score_high: float = DEFAULT_SCORING_THRESHOLDS["turnout_end_score_high"],
    turnout_drop_score_good_max: float = DEFAULT_SCORING_THRESHOLDS["turnout_drop_score_good_max"],
    turnout_drop_score_zero_at: float = DEFAULT_SCORING_THRESHOLDS["turnout_drop_score_zero_at"],
    turnout_decline_score_good_max: float = DEFAULT_SCORING_THRESHOLDS["turnout_decline_score_good_max"],
    turnout_decline_score_zero_at: float = DEFAULT_SCORING_THRESHOLDS["turnout_decline_score_zero_at"],
    turnout_outside_band_share_good_max: float = DEFAULT_SCORING_THRESHOLDS["turnout_outside_band_share_good_max"],
    turnout_outside_band_share_zero_at: float = DEFAULT_SCORING_THRESHOLDS["turnout_outside_band_share_zero_at"],
    participation_q_delta_mean_abs_good_max: float = DEFAULT_SCORING_THRESHOLDS["participation_q_delta_mean_abs_good_max"],
    participation_q_delta_mean_abs_zero_at: float = DEFAULT_SCORING_THRESHOLDS["participation_q_delta_mean_abs_zero_at"],
    participation_q_delta_late_mean_abs_good_max: float = DEFAULT_SCORING_THRESHOLDS["participation_q_delta_late_mean_abs_good_max"],
    participation_q_delta_late_mean_abs_zero_at: float = DEFAULT_SCORING_THRESHOLDS["participation_q_delta_late_mean_abs_zero_at"],
    turnout_drop_context_relief_strength: float = 0.60,
    turnout_decline_context_relief_strength: float = 0.60,
    quality_component_weights: dict[str, float] | None = None,
    return_meta: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    w = DEFAULT_SCORING_WEIGHTS if weights is None else weights
    sw = DEFAULT_STAGE_WEIGHTS if stage_weights is None else stage_weights
    df = run_features.copy()
    if "passes_hard_gates" not in df.columns:
        df = apply_hard_gates(df)

    primary = df.loc[df["rule_name"] == str(primary_rule_name)].copy()
    if len(primary) == 0:
        raise ValueError(f"No primary-rule rows found for rule_name={primary_rule_name!r}.")
    _require_columns(primary, REQUIRED_PRIMARY_SCORING_COLUMNS, context="score_designs(primary)")
    _require_numeric_notna(
        primary,
        REQUIRED_PRIMARY_SCORING_COLUMNS,
        context="score_designs(primary)",
    )
    primary["passes_hard_gates"] = pd.to_numeric(primary["passes_hard_gates"], errors="coerce").astype(bool)

    missing_optional_columns: list[str] = []
    for col in OPTIONAL_PRIMARY_SCORING_COLUMNS:
        if col not in primary.columns:
            primary[col] = np.nan
            missing_optional_columns.append(col)
    optional_na_counts = {
        col: int(pd.to_numeric(primary[col], errors="coerce").isna().sum())
        for col in OPTIONAL_PRIMARY_SCORING_COLUMNS
    }

    primary["z_turnout_std"] = _norm01(primary["turnout_std"], higher_better=True)
    primary["z_gini_std"] = _norm01(primary["gini_std"], higher_better=True)
    primary["z_dist_std"] = _norm01(primary["dist_std"], higher_better=True)
    primary["z_group_std"] = _norm01(primary["group_participation_std"], higher_better=True)
    primary["z_group_turnout_range"] = _norm01(primary["group_turnout_range_mean"], higher_better=True)
    primary["z_roll3_group_turnout_range"] = _norm01(
        primary["roll3_group_turnout_range_mean"], higher_better=True
    )
    primary["z_roll3_group_turnout_range_max"] = _norm01(
        primary["roll3_group_turnout_range_max"], higher_better=True
    )
    primary["z_roll20_group_turnout_range"] = _norm01(
        primary["roll20_group_turnout_range_mean"], higher_better=True
    )
    primary["z_roll20_group_turnout_range_max"] = _norm01(
        primary["roll20_group_turnout_range_max"], higher_better=True
    )
    primary["z_group_turnout_resid"] = _norm01(primary["group_turnout_residual_abs_mean"], higher_better=True)
    primary["z_pa_gap"] = _norm01(primary["participant_abstainer_delta_rel_gap_abs"], higher_better=True)
    primary["z_group_pa_gap"] = _norm01(primary["group_participant_abstainer_delta_rel_gap_abs"], higher_better=True)
    primary["z_winner_entropy"] = _norm01(primary["winner_entropy_norm"], higher_better=True)
    primary["z_dist_nonzero_share"] = _norm01(primary["dist_nonzero_share"], higher_better=True)
    primary["z_competitive_step_share"] = _norm01(primary["competitive_step_share"], higher_better=True)
    primary["z_winner_changes"] = _norm01(primary["winner_change_rate_post_burnin"], higher_better=True)
    primary["z_mean_turnout_centered"] = _band_pref01(
        primary["mean_turnout"],
        low=float(turnout_start_score_low),
        high=float(turnout_start_score_high),
    )
    primary["z_turnout_start_band"] = _band_pref01(
        primary["turnout_start_window_mean"],
        low=float(turnout_start_score_low),
        high=float(turnout_start_score_high),
    )
    primary["z_turnout_end_band"] = _band_pref01(
        primary["turnout_end_window_mean"],
        low=float(turnout_end_score_low),
        high=float(turnout_end_score_high),
    )
    primary["z_turnout_drop_stability_base"] = _upper_bound_pref01(
        primary["turnout_drop_start_end"],
        good_max=float(turnout_drop_score_good_max),
        zero_at=float(turnout_drop_score_zero_at),
    )
    primary["z_turnout_decline_stability_base"] = _upper_bound_pref01(
        primary["turnout_decline_slope_norm"],
        good_max=float(turnout_decline_score_good_max),
        zero_at=float(turnout_decline_score_zero_at),
    )
    entropy_good_min = float(np.clip(float(min_winner_entropy_norm) + 0.35, 0.0, 1.0))
    competitive_good_min = float(np.clip(float(min_competitive_step_share) + 0.25, 0.0, 1.0))
    primary["z_entropy_context"] = _lower_bound_pref01(
        primary["winner_entropy_norm"],
        zero_at=float(min_winner_entropy_norm),
        good_min=entropy_good_min,
    )
    primary["z_competitive_context"] = _lower_bound_pref01(
        primary["competitive_step_share"],
        zero_at=float(min_competitive_step_share),
        good_min=competitive_good_min,
    )
    primary["z_turnout_drop_context"] = primary[["z_entropy_context", "z_competitive_context"]].mean(axis=1)
    drop_relief = float(np.clip(turnout_drop_context_relief_strength, 0.0, 1.0))
    decline_relief = float(np.clip(turnout_decline_context_relief_strength, 0.0, 1.0))
    primary["z_turnout_drop_stability"] = (
        primary["z_turnout_drop_stability_base"]
        + drop_relief
        * primary["z_turnout_drop_context"]
        * (1.0 - primary["z_turnout_drop_stability_base"])
    ).clip(0.0, 1.0)
    primary["z_turnout_decline_stability"] = (
        primary["z_turnout_decline_stability_base"]
        + decline_relief
        * primary["z_turnout_drop_context"]
        * (1.0 - primary["z_turnout_decline_stability_base"])
    ).clip(0.0, 1.0)
    primary["z_turnout_band_time"] = _upper_bound_pref01(
        primary["turnout_outside_20_80_share"],
        good_max=float(turnout_outside_band_share_good_max),
        zero_at=float(turnout_outside_band_share_zero_at),
    )
    primary["z_turnout_shape"] = primary[
        [
            "z_turnout_start_band",
            # Extra weight on start-window realism to avoid top buckets with very high starts.
            "z_turnout_start_band",
            "z_turnout_end_band",
            "z_turnout_drop_stability",
            "z_turnout_decline_stability",
            "z_turnout_band_time",
        ]
    ].mean(axis=1)
    primary["z_q_delta_mean_stability"] = _upper_bound_pref01(
        pd.to_numeric(primary["participation_q_delta_mean_abs"], errors="coerce"),
        good_max=float(participation_q_delta_mean_abs_good_max),
        zero_at=float(participation_q_delta_mean_abs_zero_at),
    )
    primary["z_q_delta_late_stability"] = _upper_bound_pref01(
        pd.to_numeric(primary["participation_q_delta_late_window_mean_abs"], errors="coerce"),
        good_max=float(participation_q_delta_late_mean_abs_good_max),
        zero_at=float(participation_q_delta_late_mean_abs_zero_at),
    )
    primary["z_q_delta_group_dispersion_late"] = _norm01(
        pd.to_numeric(primary["participation_q_delta_group_dispersion_late_w"], errors="coerce"),
        higher_better=True,
    )
    primary["z_participant_share_max_drift"] = _norm01(
        pd.to_numeric(primary["participant_share_max_abs_drift_20"], errors="coerce"),
        higher_better=True,
    )
    primary["z_participant_share_mean_drift"] = _norm01(
        pd.to_numeric(primary["participant_share_mean_abs_drift_20_w"], errors="coerce"),
        higher_better=True,
    )
    primary["z_participant_share_turnover"] = _norm01(
        pd.to_numeric(primary["participant_share_turnover_rate_w"], errors="coerce"),
        higher_better=True,
    )
    primary["z_participant_composition_dynamics"] = primary[
        [
            "z_participant_share_max_drift",
            "z_participant_share_mean_drift",
            "z_participant_share_turnover",
        ]
    ].mean(axis=1)
    primary["z_puzzle_dom_balance_conflict"] = _band_pref01(
        primary["puzzle_dominance_share_conflict"],
        low=float(puzzle_dominance_share_score_low),
        high=float(puzzle_dominance_share_score_high),
    )
    quality_components = list(PRIMARY_QUALITY_COMPONENT_KEYS)
    quality_values = primary[quality_components].copy()
    if quality_component_weights is None:
        primary["run_quality_raw"] = quality_values.mean(axis=1)
        effective_qcw = {k: 1.0 / float(len(quality_components)) for k in quality_components}
    else:
        # Normalize provided weights over known components and compute row-wise
        # weighted means with NaN-robust denominator.
        w_arr = np.asarray(
            [max(0.0, float(quality_component_weights.get(k, 0.0))) for k in quality_components],
            dtype=float,
        )
        w_sum = float(np.sum(w_arr))
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            w_arr = np.full(len(quality_components), 1.0 / float(len(quality_components)), dtype=float)
        else:
            w_arr = w_arr / w_sum
        vals = quality_values.to_numpy(dtype=float)
        finite = np.isfinite(vals)
        num = np.nansum(vals * w_arr[None, :], axis=1)
        den = np.sum(finite * w_arr[None, :], axis=1)
        raw = np.divide(num, den, out=np.zeros_like(num), where=den > 0.0)
        primary["run_quality_raw"] = pd.Series(raw, index=primary.index, dtype=float)
        effective_qcw = {k: float(w_arr[i]) for i, k in enumerate(quality_components)}
    primary["run_quality_viable"] = np.where(primary["passes_hard_gates"], primary["run_quality_raw"], np.nan)

    by_design = (
        primary.groupby("design_id", as_index=False)
        .agg(
            n_primary_runs=("seed", "count"),
            n_primary_pass=("passes_hard_gates", "sum"),
            pass_rate=("passes_hard_gates", "mean"),
            quality_mean=("run_quality_viable", "mean"),
            quality_std=("run_quality_viable", "std"),
            moderate_recovery_strength_mean=("moderate_recovery_strength_run", "mean"),
        )
        .fillna({"quality_std": 0.0})
    )
    by_design["quality_mean"] = pd.to_numeric(by_design["quality_mean"], errors="coerce").fillna(0.0)
    by_design["seed_robustness"] = _norm01(by_design["quality_std"], higher_better=False)
    by_design["moderate_recovery_strength_mean"] = pd.to_numeric(
        by_design["moderate_recovery_strength_mean"], errors="coerce"
    ).fillna(0.0)
    by_design["moderate_recovery_strength_norm"] = _norm01(
        by_design["moderate_recovery_strength_mean"], higher_better=True
    )
    by_design["moderate_recovery_quality_gate"] = _lower_bound_pref01(
        by_design["quality_mean"],
        zero_at=float(moderate_selector_quality_gate_zero_at),
        good_min=float(moderate_selector_quality_gate_good_min),
    )
    by_design["moderate_recovery_selector"] = (
        by_design["moderate_recovery_strength_norm"] * by_design["moderate_recovery_quality_gate"]
    ).clip(0.0, 1.0)

    robust = df.loc[df["rule_name"] == str(robust_rule_name)].copy()
    disc_active = False
    if len(robust) > 0:
        _require_columns(robust, REQUIRED_ROBUST_SCORING_COLUMNS, context="score_designs(robust)")
        _require_numeric_notna(
            robust,
            REQUIRED_ROBUST_SCORING_COLUMNS,
            context="score_designs(robust)",
        )
        m = primary.merge(
            robust,
            on=["design_id", "seed"],
            how="inner",
            suffixes=("_p", "_r"),
        )
        if len(m) > 0:
            m["delta_turnout"] = (m["mean_turnout_p"] - m["mean_turnout_r"]).abs()
            m["delta_gini"] = (m["mean_gini_p"] - m["mean_gini_r"]).abs()
            m["delta_dist"] = (m["mean_dist_p"] - m["mean_dist_r"]).abs()
            m["delta_group_std"] = (m["group_participation_std_p"] - m["group_participation_std_r"]).abs()
            for c in ["delta_turnout", "delta_gini", "delta_dist", "delta_group_std"]:
                m[f"z_{c}"] = _norm01(m[c], higher_better=True)
            m["discriminability_seed"] = m[
                ["z_delta_turnout", "z_delta_gini", "z_delta_dist", "z_delta_group_std"]
            ].mean(axis=1)
            disc_active = True
            disc = m.groupby("design_id", as_index=False).agg(
                discriminability=("discriminability_seed", "mean"),
                n_matched_seed_pairs=("seed", "count"),
            )
        else:
            disc = pd.DataFrame(columns=["design_id", "discriminability", "n_matched_seed_pairs"])
    else:
        disc = pd.DataFrame(columns=["design_id", "discriminability", "n_matched_seed_pairs"])

    out = by_design.merge(disc, on="design_id", how="left")
    out["discriminability"] = pd.to_numeric(out["discriminability"], errors="coerce").fillna(0.0)
    out["n_matched_seed_pairs"] = pd.to_numeric(out["n_matched_seed_pairs"], errors="coerce").fillna(0).astype(int)

    w_q = float(w["quality_mean"])
    # Keep discriminability as a diagnostic only, never as a selection signal.
    w_d = 0.0
    w_r = float(w["seed_robustness"])
    w_m = float(w.get("moderate_recovery_selector", 0.0))
    denom = w_q + w_r + w_m
    if denom > 0.0:
        w_q = w_q / denom
        w_r = w_r / denom
        w_m = w_m / denom
    else:
        w_q = 0.70
        w_r = 0.30
        w_m = 0.0

    out["weight_quality_effective"] = float(w_q)
    out["weight_discriminability_effective"] = float(w_d)
    out["weight_seed_robustness_effective"] = float(w_r)
    out["weight_moderate_recovery_selector_effective"] = float(w_m)
    out["quality_bundle"] = (
        w_q * out["quality_mean"]
        + w_d * out["discriminability"]
        + w_r * out["seed_robustness"]
        + w_m * out["moderate_recovery_selector"]
    )
    s_viability = float(sw.get("viability", 0.6))
    s_quality = float(sw.get("quality_bundle", 0.4))
    denom = s_viability + s_quality
    if denom <= 0.0:
        s_viability = 0.6
        s_quality = 0.4
        denom = 1.0
    s_viability = s_viability / denom
    s_quality = s_quality / denom
    out["stage_weight_viability"] = float(s_viability)
    out["stage_weight_quality_bundle"] = float(s_quality)
    out["score_total"] = (
        s_viability * out["pass_rate"]
        + s_quality * out["quality_bundle"]
    )

    pre_filter_n = int(len(out))
    if required_primary_runs is not None:
        out = out.loc[out["n_primary_runs"] >= int(required_primary_runs)].copy()
    if required_matched_seed_pairs is not None:
        out = out.loc[out["n_matched_seed_pairs"] >= int(required_matched_seed_pairs)].copy()
    dropped_n = int(pre_filter_n - len(out))

    out = out.sort_values(["score_total", "pass_rate"], ascending=[False, False]).reset_index(drop=True)
    if return_meta:
        return out, {
            "discriminability_active": bool(disc_active),
            "effective_weights": {
                "quality_mean": float(w_q),
                "discriminability": float(w_d),
                "seed_robustness": float(w_r),
                "moderate_recovery_selector": float(w_m),
            },
            "effective_stage_weights": {
                "viability": float(s_viability),
                "quality_bundle": float(s_quality),
            },
            "effective_quality_component_weights": effective_qcw,
            "required_primary_runs": (
                None if required_primary_runs is None else int(required_primary_runs)
            ),
            "required_matched_seed_pairs": (
                None if required_matched_seed_pairs is None else int(required_matched_seed_pairs)
            ),
            "dropped_incomplete_designs": int(dropped_n),
            "optional_metric_diagnostics": {
                "missing_optional_columns": sorted(missing_optional_columns),
                "optional_na_counts": optional_na_counts,
            },
        }
    return out


def analyze_doe_root(
    doe_root: Path,
    *,
    out_dir: Path | None = None,
    burn_in_steps: int = 0,
    primary_rule_name: str = "approval",
    robust_rule_name: str = "utilitarian",
    objective_config_path: Path | None = None,
    thresholds: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
    stage_weights: dict[str, float] | None = None,
    quality_component_weights: dict[str, float] | None = None,
    strict_completeness: bool | None = None,
) -> dict[str, Path]:
    root = Path(doe_root)
    out = root if out_dir is None else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    objective_payload: dict[str, Any] | None = None
    if objective_config_path is not None:
        objective_payload = load_selection_objective(Path(objective_config_path))
        thr = dict(objective_payload["thresholds"])
        w = dict(objective_payload["weights"])
        sw = dict(objective_payload["stage_weights"])
        qcw = objective_payload.get("quality_component_weights")
        strict_flag = (
            bool(objective_payload["strict_completeness"])
            if strict_completeness is None
            else bool(strict_completeness)
        )
    else:
        thr = dict(DEFAULT_SCORING_THRESHOLDS)
        w = dict(DEFAULT_SCORING_WEIGHTS)
        sw = dict(DEFAULT_STAGE_WEIGHTS)
        qcw = None
        strict_flag = True if strict_completeness is None else bool(strict_completeness)

    if thresholds is not None:
        thr.update(dict(thresholds))
    if weights is not None:
        w.update(dict(weights))
    if stage_weights is not None:
        sw.update(dict(stage_weights))
    if quality_component_weights is not None:
        qcw = dict(quality_component_weights)

    rf = collect_run_features(
        root,
        burn_in_steps=int(burn_in_steps),
        puzzle_conflict_min_dist=float(thr["puzzle_conflict_min_dist"]),
        lockin_episode_min_len_steps=int(thr["lockin_episode_min_len_steps"]),
        lockin_recovery_window_steps=int(thr["lockin_recovery_window_steps"]),
        lockin_min_dominant_participant_share=float(thr["lockin_min_dominant_participant_share"]),
        lockin_min_altruism_share=float(thr["lockin_min_altruism_share"]),
        lockin_min_altruism_lift=float(thr["lockin_min_altruism_lift"]),
        lockin_margin_power_threshold=float(thr["lockin_margin_power_threshold"]),
        lockin_margin_puzzle_recovery_threshold=float(thr["lockin_margin_puzzle_recovery_threshold"]),
        moderate_min_group_size=int(thr["moderate_min_group_size"]),
        moderate_takeover_loss_frac_min=float(thr["moderate_takeover_loss_frac_min"]),
        moderate_takeover_capture_frac_min=float(thr["moderate_takeover_capture_frac_min"]),
        moderate_rebound_loss_frac_min=float(thr["moderate_rebound_loss_frac_min"]),
        moderate_rebound_share_vs_a_peak_min=float(thr["moderate_rebound_share_vs_a_peak_min"]),
        moderate_min_steps=int(thr["moderate_min_steps"]),
        moderate_min_peak_share=float(thr["moderate_min_peak_share"]),
        moderate_length_bonus_max=float(thr["moderate_length_bonus_max"]),
        moderate_altruism_bonus_weight=float(thr["moderate_altruism_bonus_weight"]),
        moderate_altruism_bonus_floor=float(thr["moderate_altruism_bonus_floor"]),
    )
    gated = apply_hard_gates(
        rf,
        max_all_abstain_stretch=float(thr["max_all_abstain_stretch"]),
        min_winner_changes_post_burnin=float(thr["min_winner_changes_post_burnin"]),
        min_winner_change_rate_post_burnin=float(thr["min_winner_change_rate_post_burnin"]),
        max_winner_change_rate_post_burnin=float(thr["max_winner_change_rate_post_burnin"]),
        min_group_turnout_range_mean=float(thr["min_group_turnout_range_mean"]),
        min_roll3_group_turnout_range_max=float(thr["min_roll3_group_turnout_range_max"]),
        min_roll20_group_turnout_range_max=float(thr["min_roll20_group_turnout_range_max"]),
        min_turnout_std=float(thr["min_turnout_std"]),
        min_gini_std=float(thr["min_gini_std"]),
        min_dist_std=float(thr["min_dist_std"]),
        min_winner_entropy_norm=float(thr["min_winner_entropy_norm"]),
        min_dist_nonzero_share=float(thr["min_dist_nonzero_share"]),
        min_competitive_step_share=float(thr["min_competitive_step_share"]),
        min_mean_turnout=float(thr["min_mean_turnout"]),
        max_mean_turnout=float(thr["max_mean_turnout"]),
        min_puzzle_conflict_step_share_for_gate=float(thr["min_puzzle_conflict_step_share_for_gate"]),
        max_puzzle_dominance_share_conflict=float(thr["max_puzzle_dominance_share_conflict"]),
        min_power_recovery_share_conflict=float(thr["min_power_recovery_share_conflict"]),
    )
    required_primary_runs: int | None = None
    required_matched_seed_pairs: int | None = None
    if bool(strict_flag):
        spec_path = root / "doe_spec.json"
        if spec_path.exists():
            try:
                spec = json.loads(spec_path.read_text(encoding="utf-8"))
                seeds = spec.get("seeds", [])
                if isinstance(seeds, list) and len(seeds) > 0:
                    required_primary_runs = int(len(seeds))
                include_robustness = bool(spec.get("include_robustness", False))
                robust_every = int(spec.get("robust_every", 1))
                if (
                    required_primary_runs is not None
                    and include_robustness
                    and robust_every == 1
                ):
                    required_matched_seed_pairs = int(required_primary_runs)
            except (ValueError, TypeError):
                # Keep scoring robust even when spec metadata is malformed.
                required_primary_runs = None
                required_matched_seed_pairs = None

    scores, score_meta = score_designs(
        gated,
        primary_rule_name=primary_rule_name,
        robust_rule_name=robust_rule_name,
        weights=w,
        stage_weights=sw,
        required_primary_runs=required_primary_runs,
        required_matched_seed_pairs=required_matched_seed_pairs,
        min_winner_entropy_norm=float(thr["min_winner_entropy_norm"]),
        min_competitive_step_share=float(thr["min_competitive_step_share"]),
        puzzle_dominance_share_score_low=float(thr["puzzle_dominance_share_score_low"]),
        puzzle_dominance_share_score_high=float(thr["puzzle_dominance_share_score_high"]),
        moderate_selector_quality_gate_zero_at=float(thr["moderate_selector_quality_gate_zero_at"]),
        moderate_selector_quality_gate_good_min=float(thr["moderate_selector_quality_gate_good_min"]),
        turnout_start_score_low=float(thr["turnout_start_score_low"]),
        turnout_start_score_high=float(thr["turnout_start_score_high"]),
        turnout_end_score_low=float(thr["turnout_end_score_low"]),
        turnout_end_score_high=float(thr["turnout_end_score_high"]),
        turnout_drop_score_good_max=float(thr["turnout_drop_score_good_max"]),
        turnout_drop_score_zero_at=float(thr["turnout_drop_score_zero_at"]),
        turnout_decline_score_good_max=float(thr["turnout_decline_score_good_max"]),
        turnout_decline_score_zero_at=float(thr["turnout_decline_score_zero_at"]),
        turnout_outside_band_share_good_max=float(thr["turnout_outside_band_share_good_max"]),
        turnout_outside_band_share_zero_at=float(thr["turnout_outside_band_share_zero_at"]),
        participation_q_delta_mean_abs_good_max=float(thr["participation_q_delta_mean_abs_good_max"]),
        participation_q_delta_mean_abs_zero_at=float(thr["participation_q_delta_mean_abs_zero_at"]),
        participation_q_delta_late_mean_abs_good_max=float(thr["participation_q_delta_late_mean_abs_good_max"]),
        participation_q_delta_late_mean_abs_zero_at=float(thr["participation_q_delta_late_mean_abs_zero_at"]),
        quality_component_weights=qcw,
        return_meta=True,
    )

    run_features_csv = out / "doe_run_features.csv"
    design_scores_csv = out / "doe_design_scores.csv"
    selection_spec_json = out / "doe_selection_spec.json"
    top_designs_json = out / "doe_top_designs.json"
    legacy_scoring_spec = out / "doe_scoring_spec.json"
    if legacy_scoring_spec.exists():
        legacy_scoring_spec.unlink()

    gated.to_csv(run_features_csv, index=False)
    scores.to_csv(design_scores_csv, index=False)
    selection_spec_json.write_text(
        json.dumps(
            {
                "stage": "doe_selection",
                "burn_in_steps": int(burn_in_steps),
                "primary_rule_name": str(primary_rule_name),
                "robust_rule_name": str(robust_rule_name),
                "objective_config_path": (
                    None if objective_payload is None else str(objective_payload["path"])
                ),
                "objective_config_version": (
                    None if objective_payload is None else objective_payload.get("version")
                ),
                "thresholds": thr,
                "weights": w,
                "stage_weights": sw,
                "quality_component_weights": qcw,
                "strict_completeness": bool(strict_flag),
                "effective_weights": score_meta["effective_weights"],
                "effective_stage_weights": score_meta["effective_stage_weights"],
                "effective_quality_component_weights": score_meta["effective_quality_component_weights"],
                "discriminability_active": bool(score_meta["discriminability_active"]),
                "required_primary_runs": score_meta["required_primary_runs"],
                "required_matched_seed_pairs": score_meta["required_matched_seed_pairs"],
                "dropped_incomplete_designs": int(score_meta["dropped_incomplete_designs"]),
                "optional_metric_diagnostics": score_meta["optional_metric_diagnostics"],
                "score_formula": (
                    "score_total = s_v*pass_rate + "
                    "s_q*(w_q*quality_mean + w_r*seed_robustness + "
                    "w_m*moderate_recovery_selector)"
                ),
                "quality_components": [
                    "turnout_std",
                    "gini_std",
                    "dist_std",
                    "group_participation_std",
                    "group_turnout_range_mean",
                    "roll3_group_turnout_range_mean",
                    "roll3_group_turnout_range_max",
                    "roll20_group_turnout_range_mean",
                    "roll20_group_turnout_range_max",
                    "group_turnout_residual_abs_mean",
                    "participant_abstainer_delta_rel_gap_abs",
                    "group_participant_abstainer_delta_rel_gap_abs",
                    "winner_entropy_norm",
                    "dist_nonzero_share",
                    "competitive_step_share",
                    "winner_changes_post_burnin",
                    "turnout_shape (start/end/contextual-drop/contextual-decline/band-time)",
                    "participation_q_delta_mean_abs (stability, closer to 0)",
                    "participation_q_delta_late_window_mean_abs (stability, closer to 0)",
                    "turnout_drop_start_end (context-relieved by entropy+competition; extra weight)",
                    "turnout_decline_slope_norm (context-relieved by entropy+competition; extra weight)",
                    "participation_q_delta_group_dispersion_late_w (mild, size-weighted)",
                    "participant_share_max_abs_drift_20 (mild)",
                    "participant_share_mean_abs_drift_20_w (mild, size-weighted)",
                    "participant_share_turnover_rate_w (mild, size-weighted)",
                    "puzzle_dominance_share_conflict",
                    "moderate_recovery_selector (light, quality-gated bonus; A-eats-B then B-eats-back strength)",
                ],
                "note": "burn_in_steps is an analysis warm-up exclusion window only (no simulation burn-in mutation/reset logic).",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    top_payload = {
        "top_5_designs": scores.head(5).to_dict(orient="records"),
        "num_designs": int(len(scores)),
    }
    top_designs_json.write_text(json.dumps(top_payload, indent=2), encoding="utf-8")

    return {
        "run_features_csv": run_features_csv,
        "design_scores_csv": design_scores_csv,
        "selection_spec_json": selection_spec_json,
        "top_designs_json": top_designs_json,
    }
