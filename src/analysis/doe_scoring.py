from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import itertools
import warnings

import numpy as np
import pandas as pd
import yaml

from src.analysis.quality_distance import resolve_quality_distance_series
from src.utils.ballots import score_options_c2
from src.utils.distance_functions import kendall_tau_order, spearman_fr_order
from src.utils.social_welfare_functions import (
    approval_voting,
    borda_rule,
    majority_rule,
    schulze_rule,
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
}

DEFAULT_SCORING_WEIGHTS: dict[str, float] = {
    "quality_mean": 0.70,
    "seed_robustness": 0.30,
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
    "roll3_group_turnout_range_max",
    "roll20_group_turnout_range_max",
    "participant_abstainer_delta_rel_gap_abs",
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
    "participant_share_max_abs_drift_20",
    "participant_share_mean_abs_drift_20_w",
    "participant_share_turnover_rate_w",
    "puzzle_dominance_share_conflict",
)

PRIMARY_QUALITY_COMPONENT_KEYS: tuple[str, ...] = (
    "z_quality_dist_std",
    "z_group_turnout_structure",
    "z_pa_gap",
    "z_winner_entropy",
    "z_turnout_shape",
    "z_participant_composition_dynamics",
    "z_puzzle_dom_balance_conflict",
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
    if rule_idx not in {0, 1, 2, 3, 4, 5}:
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
    rule_fns = [majority_rule, approval_voting, utilitarian_rule, borda_rule, schulze_rule, random_rule]
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
    except (TypeError, ValueError, IndexError, RuntimeError, AssertionError) as exc:
        warnings.warn(
            f"Failed power-ordering rule evaluation for rule_idx={rule_idx}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
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
    except (OSError, yaml.YAMLError, json.JSONDecodeError, TypeError, ValueError) as exc:
        warnings.warn(
            f"Failed to load run metadata for puzzle/power metrics at {run_dir}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
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


def compute_run_features_from_tables(
    area_steps: pd.DataFrame,
    agents: pd.DataFrame,
    *,
    burn_in_steps: int = 0,
    quality_target_mode: str = "reality",
) -> dict[str, float]:
    req_area = {"step", "participants", "turnout", "gini_index", "dist_to_reality", "winning_option_id"}
    req_agents = {"step", "personality_group_idx", "participating"}
    missing_area = sorted(req_area - set(area_steps.columns))
    missing_agents = sorted(req_agents - set(agents.columns))
    if missing_area:
        raise ValueError(f"area_steps missing columns: {missing_area}")
    if missing_agents:
        raise ValueError(f"agents missing columns: {missing_agents}")

    area_steps = area_steps.copy()
    area_steps["quality_distance"] = resolve_quality_distance_series(
        area_steps,
        quality_target_mode=quality_target_mode,
        run_label="doe_scoring:compute_run_features_from_tables",
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
    turnout_outside_20_80_share = float(
        np.mean((turnout_series.to_numpy(dtype=float) < 20.0) | (turnout_series.to_numpy(dtype=float) > 80.0))
    )
    turnout_decline_slope_norm = 0.0
    if n_steps >= 2:
        x = np.linspace(0.0, 1.0, num=n_steps, dtype=float)
        y = (turnout_series.to_numpy(dtype=float) / 100.0).astype(float)
        if np.isfinite(y).all():
            coeffs = np.polyfit(x, y, 1)
            turnout_decline_slope_norm = float(max(0.0, -float(coeffs[0])))

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

    roll3_group_turnout_range_max = 0.0
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
            roll20_group_turnout_range_max = float(np.nanmax(r))
    if arr.shape[0] >= 3 and arr.shape[1] >= 2:
        roll_cols = [
            np.convolve(arr[:, i], np.ones(3, dtype=float) / 3.0, mode="valid")
            for i in range(arr.shape[1])
        ]
        roll = np.vstack(roll_cols).T
        r = np.nanmax(roll, axis=1) - np.nanmin(roll, axis=1)
        if np.isfinite(r).any():
            roll3_group_turnout_range_max = float(np.nanmax(r))
    if arr.shape[0] >= 2 and arr.shape[1] >= 2:
        d = np.diff(arr, axis=0)
        valid = np.all(np.isfinite(d), axis=1)
        if valid.any():
            up = np.any(d[valid] > 1e-12, axis=1)
            down = np.any(d[valid] < -1e-12, axis=1)
            competitive_step_share = float(np.mean(up & down))

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

    return {
        "mean_turnout": float(per_step["turnout"].mean()),
        "turnout_std": _safe_std(per_step["turnout"]),
        "turnout_start_window_mean": float(turnout_start_window_mean),
        "turnout_end_window_mean": float(turnout_end_window_mean),
        "turnout_drop_start_end": float(turnout_drop_start_end),
        "turnout_outside_20_80_share": float(turnout_outside_20_80_share),
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
        "group_turnout_range_mean": float(group_turnout_range_mean),
        "roll3_group_turnout_range_max": float(roll3_group_turnout_range_max),
        "roll20_group_turnout_range_max": float(roll20_group_turnout_range_max),
        "participant_share_max_abs_drift_20": float(participant_share_max_abs_drift_20),
        "participant_share_mean_abs_drift_20_w": float(participant_share_mean_abs_drift_20_w),
        "participant_share_turnover_rate_w": float(participant_share_turnover_rate_w),
        "participant_abstainer_delta_rel_gap_abs": _gap_abs("election_delta_rel"),
    }


def collect_run_features(
    doe_root: Path,
    *,
    burn_in_steps: int = 0,
    puzzle_conflict_min_dist: float = DEFAULT_SCORING_THRESHOLDS["puzzle_conflict_min_dist"],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(doe_root.glob("design_*/rule_*/seed_*/run_0")):
        area_path = run_dir / "area_steps.parquet"
        agents_path = run_dir / "agents.parquet"
        if not area_path.exists() or not agents_path.exists():
            continue
        area = pd.read_parquet(area_path)
        agents = pd.read_parquet(agents_path)
        meta_path = run_dir / "meta.yaml"
        quality_target_mode = "reality"
        if meta_path.exists():
            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
            run_meta = meta.get("run", {}) if isinstance(meta, dict) else {}
            quality_target_mode = str(run_meta.get("quality_target_mode", "reality"))
        features = compute_run_features_from_tables(
            area,
            agents,
            burn_in_steps=burn_in_steps,
            quality_target_mode=quality_target_mode,
        )
        features.update(
            _compute_puzzle_power_metrics_for_run(
                run_dir=run_dir,
                area_steps=area,
                agents=agents,
                burn_in_steps=int(burn_in_steps),
                conflict_min_dist=float(puzzle_conflict_min_dist),
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
    weights: dict[str, float] | None = None,
    stage_weights: dict[str, float] | None = None,
    required_primary_runs: int | None = None,
    min_winner_entropy_norm: float = DEFAULT_SCORING_THRESHOLDS["min_winner_entropy_norm"],
    min_competitive_step_share: float = DEFAULT_SCORING_THRESHOLDS["min_competitive_step_share"],
    puzzle_dominance_share_score_low: float = DEFAULT_SCORING_THRESHOLDS["puzzle_dominance_share_score_low"],
    puzzle_dominance_share_score_high: float = DEFAULT_SCORING_THRESHOLDS["puzzle_dominance_share_score_high"],
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

    primary["z_quality_dist_std"] = _norm01(primary["dist_std"], higher_better=True)
    primary["z_group_turnout_structure"] = pd.concat(
        [
            _norm01(primary["group_participation_std"], higher_better=True),
            _norm01(primary["group_turnout_range_mean"], higher_better=True),
            _norm01(primary["roll3_group_turnout_range_max"], higher_better=True),
            _norm01(primary["roll20_group_turnout_range_max"], higher_better=True),
        ],
        axis=1,
    ).mean(axis=1)
    primary["z_pa_gap"] = _norm01(primary["participant_abstainer_delta_rel_gap_abs"], higher_better=True)
    primary["z_winner_entropy"] = _norm01(primary["winner_entropy_norm"], higher_better=True)
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
        )
        .fillna({"quality_std": 0.0})
    )
    by_design["quality_mean"] = pd.to_numeric(by_design["quality_mean"], errors="coerce").fillna(0.0)
    by_design["seed_robustness"] = _norm01(by_design["quality_std"], higher_better=False)

    out = by_design.copy()

    w_q = float(w["quality_mean"])
    w_r = float(w["seed_robustness"])
    denom = w_q + w_r
    if denom > 0.0:
        w_q = w_q / denom
        w_r = w_r / denom
    else:
        w_q = 0.70
        w_r = 0.30

    out["weight_quality_effective"] = float(w_q)
    out["weight_seed_robustness_effective"] = float(w_r)
    out["quality_bundle"] = (
        w_q * out["quality_mean"]
        + w_r * out["seed_robustness"]
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
    dropped_n = int(pre_filter_n - len(out))

    out = out.sort_values(["score_total", "pass_rate"], ascending=[False, False]).reset_index(drop=True)
    if return_meta:
        return out, {
            "effective_weights": {
                "quality_mean": float(w_q),
                "seed_robustness": float(w_r),
            },
            "effective_stage_weights": {
                "viability": float(s_viability),
                "quality_bundle": float(s_quality),
            },
            "effective_quality_component_weights": effective_qcw,
            "required_primary_runs": (
                None if required_primary_runs is None else int(required_primary_runs)
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
    if bool(strict_flag):
        spec_path = root / "doe_spec.json"
        if spec_path.exists():
            try:
                spec = json.loads(spec_path.read_text(encoding="utf-8"))
                seeds = spec.get("seeds", [])
                if isinstance(seeds, list) and len(seeds) > 0:
                    required_primary_runs = int(len(seeds))
            except (ValueError, TypeError):
                # Keep scoring robust even when spec metadata is malformed.
                required_primary_runs = None

    scores, score_meta = score_designs(
        gated,
        primary_rule_name=primary_rule_name,
        weights=w,
        stage_weights=sw,
        required_primary_runs=required_primary_runs,
        min_winner_entropy_norm=float(thr["min_winner_entropy_norm"]),
        min_competitive_step_share=float(thr["min_competitive_step_share"]),
        puzzle_dominance_share_score_low=float(thr["puzzle_dominance_share_score_low"]),
        puzzle_dominance_share_score_high=float(thr["puzzle_dominance_share_score_high"]),
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
                "required_primary_runs": score_meta["required_primary_runs"],
                "dropped_incomplete_designs": int(score_meta["dropped_incomplete_designs"]),
                "optional_metric_diagnostics": score_meta["optional_metric_diagnostics"],
                "score_formula": (
                    "score_total = s_v*pass_rate + "
                    "s_q*(w_q*quality_mean + w_r*seed_robustness)"
                ),
                "quality_components": list(PRIMARY_QUALITY_COMPONENT_KEYS),
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
