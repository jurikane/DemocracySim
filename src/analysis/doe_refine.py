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


def _std_effect(a: pd.Series, b: pd.Series) -> float:
    xa = pd.to_numeric(a, errors="coerce")
    xb = pd.to_numeric(b, errors="coerce")
    ma = float(xa.mean())
    mb = float(xb.mean())
    sa = float(xa.std(ddof=0))
    sb = float(xb.std(ddof=0))
    pooled = np.sqrt(max((sa * sa + sb * sb) / 2.0, 1e-12))
    return float((ma - mb) / pooled)


def build_refine_report(
    doe_root: Path,
    *,
    out_dir: Path | None = None,
    elite_fraction: float = 0.3,
    bad_fraction: float = 0.3,
    shrink_quantile: float = 0.8,
    min_keep_width_ratio: float = 0.15,
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
    merged = points.merge(scores[["design_id", "score_total", "pass_rate"]], on="design_id", how="inner")
    if len(merged) == 0:
        raise ValueError("No overlapping design_id rows between design points and design scores.")

    knobs = _select_knob_columns(points)
    if not knobs:
        raise ValueError("No numeric knob columns found in design points.")

    # Prefer passing designs; fallback to all.
    passing = merged.loc[merged["pass_rate"] > 0.0].copy()
    base = passing if len(passing) >= 3 else merged
    base = base.sort_values("score_total", ascending=False).reset_index(drop=True)
    n = len(base)
    n_elite = max(1, int(np.ceil(float(elite_fraction) * n)))
    n_bad = max(1, int(np.ceil(float(bad_fraction) * n)))
    elite = base.head(n_elite)
    bad = base.tail(n_bad)

    rows: list[dict[str, Any]] = []
    suggested_ranges: dict[str, list[float]] = {}
    for k in knobs:
        s = pd.to_numeric(base[k], errors="coerce")
        corr = float(s.corr(base["score_total"])) if s.notna().sum() >= 2 else 0.0
        eff = _std_effect(elite[k], bad[k])
        elite_s = pd.to_numeric(elite[k], errors="coerce")
        lo_q = float(elite_s.quantile((1.0 - float(shrink_quantile)) / 2.0))
        hi_q = float(elite_s.quantile(1.0 - (1.0 - float(shrink_quantile)) / 2.0))
        glob_lo = float(pd.to_numeric(merged[k], errors="coerce").min())
        glob_hi = float(pd.to_numeric(merged[k], errors="coerce").max())
        glob_w = max(glob_hi - glob_lo, 1e-12)
        min_w = float(min_keep_width_ratio) * glob_w
        if hi_q - lo_q < min_w:
            center = 0.5 * (hi_q + lo_q)
            lo_q = max(glob_lo, center - 0.5 * min_w)
            hi_q = min(glob_hi, center + 0.5 * min_w)
        rows.append(
            {
                "knob": k,
                "corr_score": corr,
                "elite_mean": float(elite_s.mean()),
                "bad_mean": float(pd.to_numeric(bad[k], errors="coerce").mean()),
                "elite_bad_std_effect": eff,
                "global_min": glob_lo,
                "global_max": glob_hi,
                "suggested_min": lo_q,
                "suggested_max": hi_q,
            }
        )
        suggested_ranges[k] = [float(lo_q), float(hi_q)]

    imp = pd.DataFrame(rows).sort_values(
        by=["elite_bad_std_effect", "corr_score"],
        key=lambda s: s.abs(),
        ascending=False,
    )

    imp_path = out / "doe_knob_importance.csv"
    sug_path = out / "doe_suggested_ranges.json"
    imp.to_csv(imp_path, index=False)
    sug_path.write_text(
        json.dumps(
            {
                "elite_fraction": float(elite_fraction),
                "bad_fraction": float(bad_fraction),
                "shrink_quantile": float(shrink_quantile),
                "min_keep_width_ratio": float(min_keep_width_ratio),
                "n_designs_used": int(n),
                "suggested_ranges": suggested_ranges,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"knob_importance_csv": imp_path, "suggested_ranges_json": sug_path}
