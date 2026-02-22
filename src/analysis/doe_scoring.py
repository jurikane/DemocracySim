from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd


DEFAULT_SCORING_THRESHOLDS: dict[str, float] = {
    "max_all_abstain_stretch": 10.0,
    "min_winner_changes_post_burnin": 3.0,
    "max_winner_changes_post_burnin": 170.0,
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
}

DEFAULT_SCORING_WEIGHTS: dict[str, float] = {
    "quality_mean": 0.45,
    "discriminability": 0.35,
    "seed_robustness": 0.20,
}

DEFAULT_STAGE_WEIGHTS: dict[str, float] = {
    "viability": 0.60,
    "quality_bundle": 0.40,
}


def load_selection_objective(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Selection objective config must be a JSON object.")
    allowed = {"version", "thresholds", "weights", "stage_weights", "strict_completeness"}
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

    return {
        "path": str(p),
        "version": payload.get("version"),
        "thresholds": thr,
        "weights": w,
        "stage_weights": sw,
        "strict_completeness": bool(strict),
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

    abstain_mask = (per_step["participants"].to_numpy(dtype=float) <= 0.0)
    max_all_abstain_stretch = _max_true_stretch(abstain_mask)

    post = per_step.loc[per_step["step"] > int(burn_in_steps), "winning_option_id"].to_numpy()
    winner_changes_post_burnin = _winner_changes(post)
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
            if not np.isfinite(w).all():
                continue
            c = np.corrcoef(w, rowvar=False)
            if c.shape[0] >= 2:
                tri = c[np.triu_indices_from(c, k=1)]
                tri = tri[np.isfinite(tri)]
                if len(tri) > 0:
                    sync_vals.append(float(np.mean(tri)))
        if sync_vals:
            roll10_group_sync_index = float(np.mean(sync_vals))

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

    lag1_group_signal_turnout_response_corr = 0.0
    if ("election_delta_rel" in agents.columns) and (len(group_step) > 0):
        sig = (
            agents.groupby(["step", "personality_group_idx"], as_index=False)
            .agg(signal=("election_delta_rel", "mean"))
            .sort_values(["personality_group_idx", "step"])
        )
        tur = group_step.rename(columns={"turnout_resident": "turnout"}).sort_values(
            ["personality_group_idx", "step"]
        )
        merged = sig.merge(tur, on=["step", "personality_group_idx"], how="inner").sort_values(
            ["personality_group_idx", "step"]
        )
        if len(merged) > 0:
            merged["turnout_next"] = merged.groupby("personality_group_idx")["turnout"].shift(-1)
            merged["turnout_delta_next"] = merged["turnout_next"] - merged["turnout"]
            valid = merged[["signal", "turnout_delta_next"]].dropna()
            if len(valid) >= 3:
                x = valid["signal"].to_numpy(dtype=float)
                y = valid["turnout_delta_next"].to_numpy(dtype=float)
                if np.std(x) > 0.0 and np.std(y) > 0.0:
                    c = float(np.corrcoef(x, y)[0, 1])
                    if np.isfinite(c):
                        lag1_group_signal_turnout_response_corr = float(c)

    return {
        "mean_turnout": float(per_step["turnout"].mean()),
        "turnout_std": _safe_std(per_step["turnout"]),
        "mean_gini": float(per_step["gini_index"].mean()),
        "gini_std": _safe_std(per_step["gini_index"]),
        "mean_dist": float(per_step["quality_distance"].mean()),
        "dist_std": _safe_std(per_step["quality_distance"]),
        "max_all_abstain_stretch": float(max_all_abstain_stretch),
        "winner_changes_post_burnin": float(winner_changes_post_burnin),
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
        "group_turnout_residual_abs_mean": float(group_turnout_residual_abs_mean),
        "participant_abstainer_delta_rel_gap_abs": _gap_abs("election_delta_rel"),
        "group_participant_abstainer_delta_rel_gap_abs": float(group_pa_delta_rel_gap_abs),
        "participant_abstainer_delta_abs_gap_abs": _gap_abs("election_delta_abs"),
        "participant_abstainer_diss_signal_gap_abs": _gap_abs("dissatisfaction_signal"),
    }


def collect_run_features(doe_root: Path, *, burn_in_steps: int = 0) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(doe_root.glob("design_*/rule_*/seed_*/run_0")):
        area_path = run_dir / "area_steps.parquet"
        agents_path = run_dir / "agents.parquet"
        if not area_path.exists() or not agents_path.exists():
            continue
        area = pd.read_parquet(area_path)
        agents = pd.read_parquet(agents_path)
        features = compute_run_features_from_tables(area, agents, burn_in_steps=burn_in_steps)

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
    max_winner_changes_post_burnin: float = DEFAULT_SCORING_THRESHOLDS["max_winner_changes_post_burnin"],
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
) -> pd.DataFrame:
    df = run_features.copy()
    if "group_turnout_range_mean" not in df.columns:
        df["group_turnout_range_mean"] = 0.0
    if "roll3_group_turnout_range_max" not in df.columns:
        df["roll3_group_turnout_range_max"] = 0.0
    if "roll20_group_turnout_range_max" not in df.columns:
        df["roll20_group_turnout_range_max"] = 0.0
    if "winner_entropy_norm" not in df.columns:
        df["winner_entropy_norm"] = 0.0
    if "dist_nonzero_share" not in df.columns:
        df["dist_nonzero_share"] = 0.0
    if "competitive_step_share" not in df.columns:
        df["competitive_step_share"] = 0.0
    if "mean_turnout" not in df.columns:
        df["mean_turnout"] = 0.0
    df["gate_no_collapse"] = df["max_all_abstain_stretch"] <= float(max_all_abstain_stretch)
    df["gate_no_lockin"] = df["winner_changes_post_burnin"] >= float(min_winner_changes_post_burnin)
    df["gate_not_too_chaotic"] = df["winner_changes_post_burnin"] <= float(max_winner_changes_post_burnin)
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
    for col in [
        "roll3_group_turnout_range_mean",
        "roll3_group_turnout_range_max",
        "roll20_group_turnout_range_mean",
        "roll20_group_turnout_range_max",
    ]:
        if col not in primary.columns:
            primary[col] = 0.0

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
    primary["z_winner_changes"] = _norm01(primary["winner_changes_post_burnin"], higher_better=True)
    primary["run_quality_raw"] = primary[
        [
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
        ]
    ].mean(axis=1)
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

    robust = df.loc[df["rule_name"] == str(robust_rule_name)].copy()
    disc_active = False
    if len(robust) > 0:
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
    w_d = float(w["discriminability"])
    w_r = float(w["seed_robustness"])
    if not disc_active:
        w_d = 0.0
        denom = w_q + w_r
        if denom > 0.0:
            w_q = w_q / denom
            w_r = w_r / denom
        else:
            w_q = 1.0
            w_r = 0.0

    out["weight_quality_effective"] = float(w_q)
    out["weight_discriminability_effective"] = float(w_d)
    out["weight_seed_robustness_effective"] = float(w_r)
    out["quality_bundle"] = (
        w_q * out["quality_mean"]
        + w_d * out["discriminability"]
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
            },
            "effective_stage_weights": {
                "viability": float(s_viability),
                "quality_bundle": float(s_quality),
            },
            "required_primary_runs": (
                None if required_primary_runs is None else int(required_primary_runs)
            ),
            "required_matched_seed_pairs": (
                None if required_matched_seed_pairs is None else int(required_matched_seed_pairs)
            ),
            "dropped_incomplete_designs": int(dropped_n),
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
        strict_flag = (
            bool(objective_payload["strict_completeness"])
            if strict_completeness is None
            else bool(strict_completeness)
        )
    else:
        thr = dict(DEFAULT_SCORING_THRESHOLDS)
        w = dict(DEFAULT_SCORING_WEIGHTS)
        sw = dict(DEFAULT_STAGE_WEIGHTS)
        strict_flag = True if strict_completeness is None else bool(strict_completeness)

    if thresholds is not None:
        thr.update(dict(thresholds))
    if weights is not None:
        w.update(dict(weights))
    if stage_weights is not None:
        sw.update(dict(stage_weights))

    rf = collect_run_features(root, burn_in_steps=int(burn_in_steps))
    gated = apply_hard_gates(
        rf,
        max_all_abstain_stretch=float(thr["max_all_abstain_stretch"]),
        min_winner_changes_post_burnin=float(thr["min_winner_changes_post_burnin"]),
        max_winner_changes_post_burnin=float(thr["max_winner_changes_post_burnin"]),
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
                "strict_completeness": bool(strict_flag),
                "effective_weights": score_meta["effective_weights"],
                "effective_stage_weights": score_meta["effective_stage_weights"],
                "discriminability_active": bool(score_meta["discriminability_active"]),
                "required_primary_runs": score_meta["required_primary_runs"],
                "required_matched_seed_pairs": score_meta["required_matched_seed_pairs"],
                "dropped_incomplete_designs": int(score_meta["dropped_incomplete_designs"]),
                "score_formula": (
                    "score_total = s_v*pass_rate + s_q*(w_q*quality_mean + "
                    "w_d*discriminability + w_r*seed_robustness)"
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
