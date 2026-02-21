from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd


def _select_knob_columns(points: pd.DataFrame) -> list[str]:
    skip = {"design_id"}
    out: list[str] = []
    for c in points.columns:
        if c in skip:
            continue
        s = pd.to_numeric(points[c], errors="coerce")
        if s.notna().any():
            out.append(c)
    return out


def _safe_norm01(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    lo = float(x.min())
    hi = float(x.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
        return pd.Series(np.zeros(len(x), dtype=float), index=x.index)
    return ((x - lo) / (hi - lo)).clip(0.0, 1.0)


def _qbin_codes(s: pd.Series, q: int = 8) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() < 4:
        return pd.Series(np.zeros(len(x), dtype=int), index=x.index)
    try:
        codes = pd.qcut(x, q=min(q, int(x.notna().sum())), labels=False, duplicates="drop")
        return pd.to_numeric(codes, errors="coerce").fillna(0).astype(int)
    except ValueError:
        return pd.Series(np.zeros(len(x), dtype=int), index=x.index)


def _eta_squared(y: pd.Series, x_codes: pd.Series) -> float:
    yy = pd.to_numeric(y, errors="coerce")
    xx = pd.to_numeric(x_codes, errors="coerce")
    valid = yy.notna() & xx.notna()
    if valid.sum() < 4:
        return 0.0
    yv = yy[valid]
    xv = xx[valid].astype(int)
    grand = float(yv.mean())
    ss_total = float(np.sum((yv - grand) ** 2))
    if ss_total <= 1e-12:
        return 0.0
    ss_between = 0.0
    for _, g in yv.groupby(xv):
        if len(g) == 0:
            continue
        mg = float(g.mean())
        ss_between += float(len(g)) * (mg - grand) ** 2
    return float(max(ss_between / ss_total, 0.0))


def _pareto_front_mask(values: np.ndarray) -> np.ndarray:
    n = values.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        vi = values[i]
        dominated = np.all(values >= vi, axis=1) & np.any(values > vi, axis=1)
        dominated[i] = False
        if np.any(dominated):
            keep[i] = False
    return keep


def _seed_fixed_effects(
    run_features: pd.DataFrame,
    points: pd.DataFrame,
    knobs: list[str],
    *,
    bootstrap_reps: int = 200,
    random_seed: int = 7,
) -> pd.DataFrame:
    cols = ["target", "knob", "coef", "ci_low", "ci_high", "n_obs"]
    if len(run_features) == 0 or "seed" not in run_features.columns:
        return pd.DataFrame(columns=cols)
    m = run_features.merge(points[["design_id", *knobs]], on="design_id", how="inner")
    if len(m) == 0:
        return pd.DataFrame(columns=cols)
    candidate_targets = [
        "passes_hard_gates",
        "mean_turnout",
        "winner_changes_post_burnin",
        "winner_entropy_norm",
        "dist_nonzero_share",
        "roll20_group_turnout_range_max",
        "group_turnout_range_mean",
    ]
    targets = [t for t in candidate_targets if t in m.columns]
    if not targets:
        return pd.DataFrame(columns=cols)

    rng = np.random.default_rng(int(random_seed))
    out_rows: list[dict[str, Any]] = []
    seed_d = pd.get_dummies(m["seed"].astype(str), prefix="seed", drop_first=True)

    Xk_raw = m[knobs].apply(pd.to_numeric, errors="coerce")
    Xk = (Xk_raw - Xk_raw.mean()) / (Xk_raw.std(ddof=0) + 1e-12)
    Xk = Xk.fillna(0.0)
    X_base = pd.concat([pd.Series(1.0, index=m.index, name="intercept"), Xk, seed_d], axis=1).to_numpy(dtype=float)
    n = X_base.shape[0]
    knob_ix = {k: 1 + i for i, k in enumerate(knobs)}

    def _fit_beta(y_arr: np.ndarray, X_arr: np.ndarray) -> np.ndarray:
        b, *_ = np.linalg.lstsq(X_arr, y_arr, rcond=None)
        return b

    for t in targets:
        y = pd.to_numeric(m[t], errors="coerce")
        if t == "passes_hard_gates":
            y = y.astype(float)
        valid = y.notna().to_numpy()
        if valid.sum() < max(8, len(knobs) + 2):
            continue
        yy = y.to_numpy(dtype=float)[valid]
        # standardize target for comparable coefficients
        yy = (yy - float(np.mean(yy))) / (float(np.std(yy)) + 1e-12)
        XX = X_base[valid]
        beta = _fit_beta(yy, XX)

        boot = np.zeros((int(bootstrap_reps), len(knobs)), dtype=float)
        n_obs = len(yy)
        for b in range(int(bootstrap_reps)):
            idx = rng.integers(0, n_obs, size=n_obs)
            bvec = _fit_beta(yy[idx], XX[idx, :])
            for j, k in enumerate(knobs):
                boot[b, j] = float(bvec[knob_ix[k]])

        for j, k in enumerate(knobs):
            out_rows.append(
                {
                    "target": str(t),
                    "knob": str(k),
                    "coef": float(beta[knob_ix[k]]),
                    "ci_low": float(np.quantile(boot[:, j], 0.05)),
                    "ci_high": float(np.quantile(boot[:, j], 0.95)),
                    "n_obs": int(n_obs),
                }
            )
    return pd.DataFrame(out_rows, columns=cols)


def _nonlinear_importance(
    run_features: pd.DataFrame,
    points: pd.DataFrame,
    scores: pd.DataFrame,
    knobs: list[str],
) -> pd.DataFrame:
    cols = ["target", "knob", "eta_squared", "n_obs"]
    base = points.merge(scores[["design_id", "score_total", "pass_rate"]], on="design_id", how="inner")
    if len(run_features) > 0:
        base = run_features.merge(base, on="design_id", how="inner")
    if len(base) == 0:
        return pd.DataFrame(columns=cols)
    candidate_targets = [
        "passes_hard_gates",
        "score_total",
        "pass_rate",
        "mean_turnout",
        "winner_changes_post_burnin",
        "winner_entropy_norm",
        "dist_nonzero_share",
        "roll20_group_turnout_range_max",
        "group_turnout_range_mean",
    ]
    targets = [t for t in candidate_targets if t in base.columns]
    rows: list[dict[str, Any]] = []
    for t in targets:
        y = pd.to_numeric(base[t], errors="coerce")
        for k in knobs:
            x = pd.to_numeric(base[k], errors="coerce")
            valid = y.notna() & x.notna()
            if int(valid.sum()) < 8:
                continue
            codes = _qbin_codes(x[valid], q=8)
            eta2 = _eta_squared(y[valid], codes)
            rows.append(
                {
                    "target": str(t),
                    "knob": str(k),
                    "eta_squared": float(eta2),
                    "n_obs": int(valid.sum()),
                }
            )
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).sort_values(["target", "eta_squared"], ascending=[True, False])


def _interaction_maps(
    run_features: pd.DataFrame,
    points: pd.DataFrame,
    scores: pd.DataFrame,
    nl_importance: pd.DataFrame,
) -> pd.DataFrame:
    cols = ["target", "knob_x", "knob_y", "x_bin", "y_bin", "cell_mean", "cell_count"]
    if len(nl_importance) == 0:
        return pd.DataFrame(columns=cols)
    base = points.merge(scores[["design_id", "score_total", "pass_rate"]], on="design_id", how="inner")
    if len(run_features) > 0:
        base = run_features.merge(base, on="design_id", how="inner")
    if len(base) == 0:
        return pd.DataFrame(columns=cols)

    top_knobs = (
        nl_importance.groupby("knob", as_index=False)["eta_squared"]
        .mean()
        .sort_values("eta_squared", ascending=False)["knob"]
        .head(4)
        .tolist()
    )
    if len(top_knobs) < 2:
        return pd.DataFrame(columns=cols)

    candidate_targets = [t for t in ["score_total", "pass_rate", "mean_turnout", "dist_nonzero_share", "roll20_group_turnout_range_max"] if t in base.columns]
    out_rows: list[dict[str, Any]] = []
    for t in candidate_targets:
        y = pd.to_numeric(base[t], errors="coerce")
        for i in range(len(top_knobs)):
            for j in range(i + 1, len(top_knobs)):
                kx = top_knobs[i]
                ky = top_knobs[j]
                x = pd.to_numeric(base[kx], errors="coerce")
                z = pd.to_numeric(base[ky], errors="coerce")
                valid = y.notna() & x.notna() & z.notna()
                if int(valid.sum()) < 20:
                    continue
                xb = _qbin_codes(x[valid], q=6)
                yb = _qbin_codes(z[valid], q=6)
                d = pd.DataFrame({"xb": xb, "yb": yb, "v": y[valid].to_numpy(dtype=float)})
                g = d.groupby(["xb", "yb"], as_index=False).agg(cell_mean=("v", "mean"), cell_count=("v", "count"))
                for _, r in g.iterrows():
                    out_rows.append(
                        {
                            "target": str(t),
                            "knob_x": str(kx),
                            "knob_y": str(ky),
                            "x_bin": int(r["xb"]),
                            "y_bin": int(r["yb"]),
                            "cell_mean": float(r["cell_mean"]),
                            "cell_count": int(r["cell_count"]),
                        }
                    )
    return pd.DataFrame(out_rows, columns=cols)


def _bootstrap_design_ci(
    run_features: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    bootstrap_reps: int = 500,
    random_seed: int = 11,
) -> pd.DataFrame:
    cols = ["design_id", "score_proxy_mean", "ci_low", "ci_high", "n_runs"]
    if len(run_features) == 0:
        # fallback: deterministic CI from score_total
        if "score_total" in scores.columns:
            d = scores.copy()
            d["score_proxy_mean"] = pd.to_numeric(d["score_total"], errors="coerce").fillna(0.0)
            d["ci_low"] = d["score_proxy_mean"]
            d["ci_high"] = d["score_proxy_mean"]
            d["n_runs"] = 1
            return d[["design_id", "score_proxy_mean", "ci_low", "ci_high", "n_runs"]]
        return pd.DataFrame(columns=cols)

    rf = run_features.copy()
    proxy_parts = []
    for c in [
        "winner_entropy_norm",
        "dist_nonzero_share",
        "roll20_group_turnout_range_max",
        "group_turnout_range_mean",
        "competitive_step_share",
    ]:
        if c in rf.columns:
            proxy_parts.append(_safe_norm01(rf[c]))
    if not proxy_parts:
        rf["score_proxy"] = 0.0
    else:
        rf["score_proxy"] = pd.concat(proxy_parts, axis=1).mean(axis=1)

    rng = np.random.default_rng(int(random_seed))
    out_rows: list[dict[str, Any]] = []
    for did, g in rf.groupby("design_id"):
        vals = pd.to_numeric(g["score_proxy"], errors="coerce").dropna().to_numpy(dtype=float)
        n = len(vals)
        if n == 0:
            continue
        boots = np.zeros(int(bootstrap_reps), dtype=float)
        for b in range(int(bootstrap_reps)):
            idx = rng.integers(0, n, size=n)
            boots[b] = float(np.mean(vals[idx]))
        out_rows.append(
            {
                "design_id": int(did),
                "score_proxy_mean": float(np.mean(vals)),
                "ci_low": float(np.quantile(boots, 0.05)),
                "ci_high": float(np.quantile(boots, 0.95)),
                "n_runs": int(n),
            }
        )
    return pd.DataFrame(out_rows, columns=cols).sort_values("score_proxy_mean", ascending=False)


def _pareto_designs(scores: pd.DataFrame) -> pd.DataFrame:
    cols = ["design_id", "is_pareto", "pass_rate", "quality_mean", "discriminability", "seed_robustness", "score_total"]
    if len(scores) == 0:
        return pd.DataFrame(columns=cols)
    work = scores.copy()
    for c in ["pass_rate", "quality_mean", "discriminability", "seed_robustness", "score_total"]:
        if c not in work.columns:
            work[c] = 0.0
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0.0)
    objectives = work[["pass_rate", "quality_mean", "discriminability", "seed_robustness"]].to_numpy(dtype=float)
    mask = _pareto_front_mask(objectives)
    out = work[["design_id", "pass_rate", "quality_mean", "discriminability", "seed_robustness", "score_total"]].copy()
    out["is_pareto"] = mask.astype(bool)
    return out[cols].sort_values(["is_pareto", "score_total"], ascending=[False, False])


def build_inference_report(
    doe_root: Path,
    *,
    out_dir: Path | None = None,
    bootstrap_reps: int = 500,
    random_seed: int = 11,
) -> dict[str, Path]:
    root = Path(doe_root)
    out = root if out_dir is None else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pts_path = root / "doe_design_points.csv"
    scores_path = root / "doe_design_scores.csv"
    if not pts_path.exists() or not scores_path.exists():
        raise FileNotFoundError("Expected doe_design_points.csv and doe_design_scores.csv in DOE root.")

    points = pd.read_csv(pts_path)
    scores = pd.read_csv(scores_path)
    run_features_path = root / "doe_run_features.csv"
    run_features = pd.read_csv(run_features_path) if run_features_path.exists() else pd.DataFrame()
    overlap = points.merge(scores[["design_id"]], on="design_id", how="inner")
    if len(overlap) == 0:
        raise ValueError("No overlapping design_id rows between design points and design scores.")

    knobs = _select_knob_columns(points)
    if not knobs:
        raise ValueError("No numeric knob columns found in design points.")

    spec_path = out / "doe_inference_spec.json"
    seedfx_path = out / "doe_seed_fixed_effects.csv"
    nonlin_path = out / "doe_nonlinear_importance.csv"
    inter_path = out / "doe_interaction_maps.csv"
    boot_path = out / "doe_bootstrap_design_ci.csv"
    pareto_path = out / "doe_pareto_designs.csv"
    legacy_paths = [
        out / "doe_knob_importance.csv",
        out / "doe_suggested_ranges.json",
        out / "doe_inference_summary.json",
    ]
    for p in legacy_paths:
        if p.exists():
            p.unlink()

    seed_fx = _seed_fixed_effects(
        run_features,
        points,
        knobs,
        bootstrap_reps=int(bootstrap_reps),
        random_seed=int(random_seed),
    )
    nonlin = _nonlinear_importance(run_features, points, scores, knobs)
    inter = _interaction_maps(run_features, points, scores, nonlin)
    boot = _bootstrap_design_ci(
        run_features,
        scores,
        bootstrap_reps=int(bootstrap_reps),
        random_seed=int(random_seed),
    )
    pareto = _pareto_designs(scores)

    seed_fx.to_csv(seedfx_path, index=False)
    nonlin.to_csv(nonlin_path, index=False)
    inter.to_csv(inter_path, index=False)
    boot.to_csv(boot_path, index=False)
    pareto.to_csv(pareto_path, index=False)
    spec_path.write_text(
        json.dumps(
            {
                "stage": "doe_inference",
                "bootstrap_reps": int(bootstrap_reps),
                "random_seed": int(random_seed),
                "n_design_points": int(len(points)),
                "n_design_scores": int(len(scores)),
                "n_run_features": int(len(run_features)),
                "n_knobs": int(len(knobs)),
                "methods": {
                    "seed_fixed_effects": {
                        "kind": "seed-fixed-effects OLS with bootstrap CIs",
                        "note": "Mixed-effects replacement when statsmodels is unavailable in runtime.",
                        "rows": int(len(seed_fx)),
                    },
                    "nonlinear_importance": {
                        "kind": "quantile-bin eta-squared",
                        "rows": int(len(nonlin)),
                    },
                    "interaction_maps": {
                        "kind": "2D quantile-bin cell means for top nonlinear knobs",
                        "rows": int(len(inter)),
                    },
                    "bootstrap_design_ci": {
                        "kind": "design-level score proxy bootstrap CI",
                        "rows": int(len(boot)),
                    },
                    "pareto": {
                        "kind": "non-dominated designs on pass/quality/discriminability/robustness",
                        "pareto_count": int(pareto["is_pareto"].sum()) if "is_pareto" in pareto.columns else 0,
                    },
                },
                "artifacts": {
                    "seed_fixed_effects_csv": str(seedfx_path),
                    "nonlinear_importance_csv": str(nonlin_path),
                    "interaction_maps_csv": str(inter_path),
                    "bootstrap_design_ci_csv": str(boot_path),
                    "pareto_designs_csv": str(pareto_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "inference_spec_json": spec_path,
        "seed_fixed_effects_csv": seedfx_path,
        "nonlinear_importance_csv": nonlin_path,
        "interaction_maps_csv": inter_path,
        "bootstrap_design_ci_csv": boot_path,
        "pareto_designs_csv": pareto_path,
    }
