from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.analysis.summary_render_area import _draw_area_personality_group_distribution
from src.analysis.summary_tooling import SUMMARY_PROFILE_DEBUG_DOE_COMPACT, generate_run_summary_batch2


DEFAULT_BUNDLE_DIRNAME = "doe_score_output"


@dataclass(frozen=True)
class DOEReviewBundleArtifacts:
    bundle_root: Path
    queue_csv: Path
    bucket_manifest_csv: Path
    knob_ranges_csv: Path
    knob_correlations_csv: Path
    knob_effects_csv: Path
    analysis_summary_pdf: Path


def _safe_float(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    return f


def _resolve_doe_root(raw: str | Path) -> Path:
    p = Path(str(raw))
    if p.exists():
        return p
    candidate = Path("data") / "simulation_output" / str(raw)
    if candidate.exists():
        return candidate
    return p


def _sample_buckets(design_df: pd.DataFrame, *, n_per_bucket: int) -> dict[str, pd.DataFrame]:
    ranked = design_df.sort_values(["score_total", "pass_rate", "design_id"], ascending=[False, False, True]).reset_index(drop=True)
    n = len(ranked)
    if n == 0:
        return {"top": ranked, "mid": ranked, "bottom": ranked}

    k = min(int(max(1, n_per_bucket)), n)
    top = ranked.head(k).copy()
    bottom = ranked.tail(k).sort_values(["score_total", "pass_rate", "design_id"], ascending=[True, True, True]).copy()

    viable = ranked[
        (pd.to_numeric(ranked.get("score_total"), errors="coerce").fillna(0.0) > 0.0)
        | (pd.to_numeric(ranked.get("pass_rate"), errors="coerce").fillna(0.0) > 0.0)
    ].copy()
    mid_pool = viable.reset_index(drop=True) if len(viable) >= k else ranked
    m = len(mid_pool)
    mid_center = m // 2
    half = k // 2
    start = max(0, mid_center - half)
    end = min(m, start + k)
    start = max(0, end - k)
    mid = mid_pool.iloc[start:end].copy()
    return {"top": top, "mid": mid, "bottom": bottom}


def _pick_representative_run(run_df: pd.DataFrame, *, bucket: str) -> pd.Series:
    df = run_df.copy()
    if bucket == "bottom":
        sort_cols = [
            "passes_hard_gates",
            "score_total",
            "roll20_group_turnout_range_max",
            "competitive_step_share",
            "winner_entropy_norm",
        ]
        asc = [True, True, True, True, True]
    else:
        sort_cols = [
            "passes_hard_gates",
            "score_total",
            "roll20_group_turnout_range_max",
            "competitive_step_share",
            "winner_entropy_norm",
        ]
        asc = [False, False, False, False, False]

    cols = [c for c in sort_cols if c in df.columns]
    asc_eff = [asc[sort_cols.index(c)] for c in cols]
    if "seed" in df.columns:
        cols = cols + ["seed"]
        asc_eff = asc_eff + [True]
    if cols:
        df = df.sort_values(cols, ascending=asc_eff, kind="mergesort")
    return df.iloc[0]


def _load_doe_knob_ranges(*, doe_root: Path, knob_cols: list[str], points_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    spec_path = doe_root / "doe_spec.json"
    spec = {}
    if spec_path.exists():
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            warnings.warn(
                f"Could not parse {spec_path}; falling back to observed design-point ranges: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            spec = {}
    spec_ranges = spec.get("ranges") if isinstance(spec, dict) else None
    if isinstance(spec_ranges, dict):
        for k, v in spec_ranges.items():
            if k in knob_cols and isinstance(v, list) and len(v) >= 2:
                lo = _safe_float(v[0])
                hi = _safe_float(v[1])
                if np.isfinite(lo) and np.isfinite(hi) and lo <= hi:
                    ranges[str(k)] = (float(lo), float(hi))
    for c in knob_cols:
        if c in ranges:
            continue
        vals = pd.to_numeric(points_df[c], errors="coerce")
        lo = float(vals.min()) if vals.notna().any() else float("nan")
        hi = float(vals.max()) if vals.notna().any() else float("nan")
        ranges[c] = (lo, hi)
    return ranges


def _normalized_in_range(v: float, lo: float, hi: float) -> float:
    if not np.isfinite(v) or not np.isfinite(lo) or not np.isfinite(hi):
        return float("nan")
    if hi <= lo:
        return 0.5
    x = (float(v) - float(lo)) / (float(hi) - float(lo))
    return float(max(0.0, min(1.0, x)))


def _build_knob_position_payload(
    *,
    knob_cols: list[str],
    knob_ranges: dict[str, tuple[float, float]],
    design_row: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for knob in knob_cols:
        lo, hi = knob_ranges.get(knob, (float("nan"), float("nan")))
        val = _safe_float(design_row.get(knob))
        rows.append(
            {
                "knob": knob,
                "value": val,
                "range_min": lo,
                "range_max": hi,
                "pos_0_1": _normalized_in_range(val, lo, hi),
            }
        )
    return pd.DataFrame(rows)


def _load_personality_group_info(run_dir: Path) -> dict[str, Any]:
    static_path = run_dir / "static.json"
    if not static_path.exists():
        raise FileNotFoundError(f"Missing required artifact for DOE review bundle packet: {static_path}")
    try:
        static = json.loads(static_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Failed to parse required artifact {static_path}: {exc}") from exc
    info = static.get("personality_group_info")
    if not isinstance(info, dict):
        raise RuntimeError(
            f"Invalid or missing personality_group_info in required artifact {static_path}"
        )
    return info


def _render_run_overview_pdf(
    *,
    out_pdf: Path,
    run_dir: Path,
    bucket: str,
    rank_in_bucket: int,
    design_row: pd.Series,
    run_row: pd.Series,
    knob_payload: pd.DataFrame,
) -> None:
    pg_info = _load_personality_group_info(run_dir)
    personality_groups = np.asarray(pg_info.get("personality_groups", []), dtype=int)
    global_dist = np.asarray(pg_info.get("global_distribution", []), dtype=float)
    area_dist_map = pg_info.get("areas", {}) if isinstance(pg_info.get("areas"), dict) else {}
    num_colors = int(personality_groups.shape[1]) if personality_groups.ndim == 2 and personality_groups.size > 0 else 0

    fig = plt.figure(figsize=(11.69, 8.27), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.15], width_ratios=[1.15, 1.45], hspace=0.26, wspace=0.20)
    ax_meta = fig.add_subplot(gs[0, 0])
    ax_knobs = fig.add_subplot(gs[0, 1])
    ax_pg_global = fig.add_subplot(gs[1, 0])
    ax_pg_area = fig.add_subplot(gs[1, 1])

    ax_meta.axis("off")
    design_id = int(_safe_float(design_row.get("design_id")))
    seed = int(_safe_float(run_row.get("seed"))) if np.isfinite(_safe_float(run_row.get("seed"))) else -1
    gate_flag = bool(run_row.get("gate_puzzle_anti_monopoly", True))
    lines = [
        f"DOE run packet",
        f"bucket: {bucket}  rank: {int(rank_in_bucket)}",
        f"design_id: {design_id}",
        f"seed: {seed}",
        f"rule: {str(run_row.get('rule_name', 'n/a'))}",
        f"score_total: {_safe_float(design_row.get('score_total')):.4f}",
        f"pass_rate: {_safe_float(design_row.get('pass_rate')):.4f}",
        f"quality_mean: {_safe_float(design_row.get('quality_mean')):.4f}",
        f"gate_puzzle_anti_monopoly: {'PASS' if gate_flag else 'FAIL'}",
        "",
        f"run_dir:",
        f"{run_dir}",
    ]
    ax_meta.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9)

    ax_knobs.set_title("Knob Settings in DOE Range")
    ax_knobs.set_facecolor("#f8f9fc")
    if knob_payload.empty:
        ax_knobs.text(0.5, 0.5, "No knob payload", ha="center", va="center")
        ax_knobs.set_yticks([])
        ax_knobs.set_xticks([])
    else:
        y = np.arange(len(knob_payload), dtype=float)
        pos = knob_payload["pos_0_1"].to_numpy(dtype=float)
        val = knob_payload["value"].to_numpy(dtype=float)
        lo = knob_payload["range_min"].to_numpy(dtype=float)
        hi = knob_payload["range_max"].to_numpy(dtype=float)
        colors = plt.cm.Blues(np.linspace(0.55, 0.88, max(1, len(knob_payload))))
        for i in range(len(knob_payload)):
            ax_knobs.hlines(y[i], 0.0, 1.0, color="#d6dbe6", linewidth=8.0, alpha=0.9, zorder=1)
            if np.isfinite(pos[i]):
                ax_knobs.hlines(y[i], 0.0, float(pos[i]), color=colors[i], linewidth=8.0, alpha=0.38, zorder=2)
                ax_knobs.scatter(
                    float(pos[i]),
                    y[i],
                    color=colors[i],
                    edgecolors="white",
                    linewidth=0.9,
                    s=64,
                    zorder=4,
                )
            txt = f"{val[i]:.4g} [{lo[i]:.4g}, {hi[i]:.4g}]"
            ax_knobs.text(1.02, y[i], txt, va="center", ha="left", fontsize=7.5, color="#2f2f2f")
        ax_knobs.set_xlim(0.0, 1.28)
        ax_knobs.set_yticks(y)
        ax_knobs.set_yticklabels(knob_payload["knob"].tolist(), fontsize=8)
        ax_knobs.set_xlabel("normalized position in DOE range")
        ax_knobs.grid(True, axis="x", alpha=0.20, linestyle=":")
        ax_knobs.invert_yaxis()
    _draw_area_personality_group_distribution(
        ax=ax_pg_global,
        pg_dist=global_dist,
        personality_groups=personality_groups,
        num_colors=int(num_colors),
    )
    ax_pg_global.set_title("Preference Group Dists (global)")

    if area_dist_map:
        area_candidates = []
        for area_key, payload in area_dist_map.items():
            if not isinstance(payload, dict):
                continue
            dist_vals = np.asarray(payload.get("personality_group_distribution", []), dtype=float)
            area_n = int(payload.get("num_agents", 0))
            area_candidates.append((int(area_key), area_n, dist_vals))
        area_candidates = [row for row in area_candidates if row[2].size > 0]
        if area_candidates:
            area_candidates.sort(key=lambda x: (x[1], -x[0]), reverse=True)
            area_id, area_n, area_dist = area_candidates[0]
            _draw_area_personality_group_distribution(
                ax=ax_pg_area,
                pg_dist=area_dist,
                personality_groups=personality_groups,
                num_colors=int(num_colors),
            )
            ax_pg_area.set_title(f"Preference Group Dists (area {area_id}, n={area_n})")
        else:
            ax_pg_area.axis("off")
            ax_pg_area.text(0.5, 0.5, "No area group distribution metadata", ha="center", va="center")
    else:
        ax_pg_area.axis("off")
        ax_pg_area.text(0.5, 0.5, "No area group distribution metadata", ha="center", va="center")

    fig.suptitle(f"DOE Run Overview | bucket={bucket} | design={design_id} | seed={seed}", fontsize=11)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="constrained_layout not applied because axes sizes collapsed to zero.*",
                category=UserWarning,
            )
            pdf.savefig(fig, dpi=140)
    plt.close(fig)


def _build_knob_metric_artifacts(
    *,
    points_df: pd.DataFrame,
    design_scores_df: pd.DataFrame,
    run_features_df: pd.DataFrame,
    knob_cols: list[str],
    knob_ranges: dict[str, tuple[float, float]],
    out_dir: Path,
    rule_name: str,
) -> tuple[Path, Path, Path]:
    agg_cols = [
        "mean_turnout",
        "mean_dist",
        "winner_entropy_norm",
        "competitive_step_share",
        "roll20_group_turnout_range_max",
        "puzzle_dominance_share_conflict",
        "power_recovery_share_conflict",
    ]
    agg_map = {c: (c, "mean") for c in agg_cols if c in run_features_df.columns}
    if agg_map:
        agg = run_features_df.groupby("design_id", as_index=False).agg(**agg_map)
    else:
        agg = pd.DataFrame({"design_id": sorted(int(v) for v in run_features_df["design_id"].dropna().unique().tolist())})
    base = points_df.merge(design_scores_df, on="design_id", how="inner")
    base = base.merge(agg, on="design_id", how="left")
    metrics = [
        "score_total",
        "pass_rate",
        "quality_mean",
        "seed_robustness",
        "mean_turnout",
        "mean_dist",
        "winner_entropy_norm",
        "competitive_step_share",
        "roll20_group_turnout_range_max",
        "puzzle_dominance_share_conflict",
        "power_recovery_share_conflict",
    ]
    metrics = [m for m in metrics if m in base.columns]

    corr_rows: list[dict[str, Any]] = []
    for knob in knob_cols:
        x = pd.to_numeric(base[knob], errors="coerce")
        for metric in metrics:
            y = pd.to_numeric(base[metric], errors="coerce")
            mask = x.notna() & y.notna()
            if int(mask.sum()) < 3:
                rho = float("nan")
            else:
                rho = float(x[mask].corr(y[mask], method="spearman"))
            corr_rows.append(
                {
                    "knob": knob,
                    "metric": metric,
                    "spearman_rho": rho,
                    "n": int(mask.sum()),
                }
            )
    corr_df = pd.DataFrame(corr_rows)
    corr_csv = out_dir / "knob_metric_correlations.csv"
    corr_df.to_csv(corr_csv, index=False)

    effects_rows: list[dict[str, Any]] = []
    score_vals = pd.to_numeric(base.get("score_total"), errors="coerce")
    if score_vals.notna().any():
        score_threshold = float(score_vals.quantile(0.90))
    else:
        score_threshold = float("nan")
    top_mask = score_vals >= score_threshold if np.isfinite(score_threshold) else pd.Series(False, index=base.index)
    best_idx = int(score_vals.idxmax()) if score_vals.notna().any() else None

    for knob in knob_cols:
        lo, hi = knob_ranges.get(knob, (float("nan"), float("nan")))
        x = pd.to_numeric(base[knob], errors="coerce")
        rho_row = corr_df[(corr_df["knob"] == knob) & (corr_df["metric"] == "score_total")]
        rho_score = float(rho_row["spearman_rho"].iloc[0]) if len(rho_row) > 0 else float("nan")
        if best_idx is None:
            best_value = float("nan")
        else:
            best_value = _safe_float(base.loc[best_idx, knob])
        if bool(top_mask.any()):
            top_vals = x[top_mask].dropna().to_numpy(dtype=float)
            q25 = float(np.quantile(top_vals, 0.25)) if top_vals.size > 0 else float("nan")
            q75 = float(np.quantile(top_vals, 0.75)) if top_vals.size > 0 else float("nan")
        else:
            q25 = float("nan")
            q75 = float("nan")
        effects_rows.append(
            {
                "knob": knob,
                "score_rho": rho_score,
                "impact_abs_rho": abs(rho_score) if np.isfinite(rho_score) else float("nan"),
                "best_value_at_max_score": best_value,
                "best_pos_0_1": _normalized_in_range(best_value, lo, hi),
                "top10_q25_value": q25,
                "top10_q75_value": q75,
                "top10_q25_pos_0_1": _normalized_in_range(q25, lo, hi),
                "top10_q75_pos_0_1": _normalized_in_range(q75, lo, hi),
                "range_min": lo,
                "range_max": hi,
            }
        )
    effects_df = pd.DataFrame(effects_rows)
    effects_csv = out_dir / "knob_effect_summary.csv"
    effects_df.to_csv(effects_csv, index=False)

    summary_pdf = out_dir / "doe_analysis_summary.pdf"
    with PdfPages(summary_pdf) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27), gridspec_kw={"width_ratios": [1.25, 1.0]})
        ax0, ax1 = axes
        pivot = corr_df.pivot(index="knob", columns="metric", values="spearman_rho")
        if not pivot.empty:
            mat = pivot.to_numpy(dtype=float)
            im = ax0.imshow(mat, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
            ax0.set_xticks(np.arange(pivot.shape[1]))
            ax0.set_xticklabels(pivot.columns.tolist(), rotation=45, ha="right", fontsize=8)
            ax0.set_yticks(np.arange(pivot.shape[0]))
            ax0.set_yticklabels(pivot.index.tolist(), fontsize=8)
            ax0.set_title("Knob ↔ Metric Spearman Correlations")
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    v = mat[i, j]
                    if np.isfinite(v):
                        ax0.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="black")
            cbar = fig.colorbar(im, ax=ax0, shrink=0.8)
            cbar.set_label("rho")
        else:
            ax0.axis("off")
            ax0.text(0.5, 0.5, "No correlation data", ha="center", va="center")

        if not effects_df.empty:
            idx = np.arange(len(effects_df), dtype=float)
            impact = effects_df["impact_abs_rho"].to_numpy(dtype=float)
            best_pos = effects_df["best_pos_0_1"].to_numpy(dtype=float)
            q25 = effects_df["top10_q25_pos_0_1"].to_numpy(dtype=float)
            q75 = effects_df["top10_q75_pos_0_1"].to_numpy(dtype=float)
            ax1.barh(idx, impact, color="#5fa2dd", alpha=0.8, label="impact |rho(score_total)|")
            ax1.scatter(best_pos, idx, color="#d62728", s=24, zorder=3, label="optimum pos @ max score")
            for i in range(len(idx)):
                if np.isfinite(q25[i]) and np.isfinite(q75[i]):
                    ax1.plot([q25[i], q75[i]], [idx[i], idx[i]], color="#2ca02c", linewidth=2.0, alpha=0.85)
            ax1.set_yticks(idx)
            ax1.set_yticklabels(effects_df["knob"].tolist(), fontsize=8)
            ax1.set_xlim(0.0, 1.0)
            ax1.set_xlabel("normalized scale [0..1]")
            ax1.set_title("Knob Impact + Optimum Position (top-10% band in green)")
            ax1.grid(True, axis="x", alpha=0.25)
            ax1.legend(loc="lower right", fontsize=7)
            ax1.invert_yaxis()
        else:
            ax1.axis("off")
            ax1.text(0.5, 0.5, "No effect summary data", ha="center", va="center")

        fig.suptitle(f"DOE Analysis Summary | rule={rule_name}", fontsize=12)
        fig.tight_layout()
        pdf.savefig(fig, dpi=140)
        plt.close(fig)

    return corr_csv, effects_csv, summary_pdf


def _copy_scoring_sidecars(*, doe_root: Path, out_dir: Path) -> None:
    for name in (
        "doe_design_scores.csv",
        "doe_run_features.csv",
        "doe_design_points.csv",
        "doe_selection_spec.json",
        "doe_spec.json",
        "doe_seed_selection.json",
    ):
        src = doe_root / name
        if not src.exists():
            continue
        shutil.copy2(src, out_dir / name)


def build_doe_review_bundle(
    *,
    doe_root: Path | str,
    out_dir: Path | None = None,
    per_bucket: int = 3,
    rule_name: str = "approval",
    summary_profile: str = SUMMARY_PROFILE_DEBUG_DOE_COMPACT,
    render_summaries: bool = True,
) -> DOEReviewBundleArtifacts:
    root = _resolve_doe_root(doe_root)
    if not root.exists():
        raise FileNotFoundError(f"DOE root not found: {root}")

    design_scores_path = root / "doe_design_scores.csv"
    run_features_path = root / "doe_run_features.csv"
    design_points_path = root / "doe_design_points.csv"
    if not design_scores_path.exists() or not run_features_path.exists() or not design_points_path.exists():
        raise FileNotFoundError(
            "DOE root must contain doe_design_scores.csv, doe_run_features.csv and doe_design_points.csv."
        )

    bundle_root = (root / DEFAULT_BUNDLE_DIRNAME) if out_dir is None else Path(out_dir)
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True, exist_ok=True)
    raw_dir = bundle_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    design_scores = pd.read_csv(design_scores_path)
    run_features = pd.read_csv(run_features_path)
    design_points = pd.read_csv(design_points_path)
    if "rule_name" in run_features.columns:
        run_features = run_features[run_features["rule_name"].astype(str) == str(rule_name)].copy()
    if run_features.empty:
        raise RuntimeError(f"No run features found for rule_name={rule_name!r}")

    if design_scores.empty:
        raise RuntimeError(
            "doe_design_scores.csv is empty. Run scripts/score_doe.py first "
            "(it infers primary/robust rules from doe_spec.json)."
        )

    base_scores = design_points.merge(design_scores, on="design_id", how="inner")
    if base_scores.empty:
        raise RuntimeError(
            "doe_design_scores.csv could not be merged to doe_design_points.csv by design_id. "
            "Run scripts/score_doe.py to regenerate scoring outputs."
        )
    buckets = _sample_buckets(base_scores, n_per_bucket=int(per_bucket))
    knob_cols = [c for c in design_points.columns if c != "design_id"]
    knob_ranges = _load_doe_knob_ranges(doe_root=root, knob_cols=knob_cols, points_df=design_points)

    _copy_scoring_sidecars(doe_root=root, out_dir=raw_dir)

    knob_ranges_rows = []
    for k in knob_cols:
        lo, hi = knob_ranges.get(k, (float("nan"), float("nan")))
        knob_ranges_rows.append({"knob": k, "range_min": lo, "range_max": hi})
    knob_ranges_csv = raw_dir / "knob_ranges.csv"
    pd.DataFrame(knob_ranges_rows).to_csv(knob_ranges_csv, index=False)
    queue_rows: list[dict[str, Any]] = []
    queue_cols = [
        "doe_root",
        "bucket",
        "rank_in_bucket",
        "design_id",
        "score_total",
        "pass_rate",
        "rule_name",
        "seed",
        "representative_run_dir",
        "bundle_packet_dir",
        "bundle_raw_packet_dir",
        "bundle_overview_pdf",
        "bundle_used_fallback_scores",
    ]
    packet_rows: list[dict[str, Any]] = []

    for bucket_name in ("top", "mid", "bottom"):
        bucket_dir = bundle_root / bucket_name
        bucket_dir.mkdir(parents=True, exist_ok=True)
        bucket_df = buckets[bucket_name].reset_index(drop=True)
        for i, (_, drow) in enumerate(bucket_df.iterrows(), start=1):
            did = int(drow["design_id"])
            runs = run_features[run_features["design_id"].astype(int) == did]
            if runs.empty:
                continue
            rep = _pick_representative_run(runs, bucket=bucket_name)
            run_dir = Path(str(rep.get("run_dir", "")))
            if not run_dir.exists():
                continue
            seed = int(_safe_float(rep.get("seed"))) if np.isfinite(_safe_float(rep.get("seed"))) else -1
            packet_name = f"{i:02d}_design_{did:04d}_seed_{seed:05d}"
            packet_dir = bucket_dir / packet_name
            packet_dir.mkdir(parents=True, exist_ok=True)
            raw_packet_dir = raw_dir / "packets" / bucket_name / packet_name
            raw_packet_dir.mkdir(parents=True, exist_ok=True)

            summary_out_dir = raw_packet_dir / "summary"
            if render_summaries:
                generate_run_summary_batch2(
                    run_dir=run_dir,
                    out_dir=summary_out_dir,
                    profile=summary_profile,
                )
                for pdf_path in summary_out_dir.glob("*.pdf"):
                    shutil.copy2(pdf_path, packet_dir / pdf_path.name)
                    # Keep raw folder focused on non-PDF sidecars to avoid duplicate browsing targets.
                    pdf_path.unlink(missing_ok=True)

            knob_payload = _build_knob_position_payload(
                knob_cols=knob_cols,
                knob_ranges=knob_ranges,
                design_row=drow,
            )
            run_overview_pdf = packet_dir / "run_overview.pdf"
            _render_run_overview_pdf(
                out_pdf=run_overview_pdf,
                run_dir=run_dir,
                bucket=bucket_name,
                rank_in_bucket=i,
                design_row=drow,
                run_row=rep,
                knob_payload=knob_payload,
            )
            knob_payload.to_csv(raw_packet_dir / "knob_profile.csv", index=False)

            packet_meta = {
                "bucket": bucket_name,
                "rank_in_bucket": i,
                "design_id": did,
                "seed": seed,
                "rule_name": str(rep.get("rule_name", "")),
                "score_total": _safe_float(drow.get("score_total")),
                "pass_rate": _safe_float(drow.get("pass_rate")),
                "quality_mean": _safe_float(drow.get("quality_mean")),
                "seed_robustness": _safe_float(drow.get("seed_robustness")),
                "gate_puzzle_anti_monopoly": bool(rep.get("gate_puzzle_anti_monopoly", True)),
                "passes_hard_gates": bool(rep.get("passes_hard_gates", False)),
                "bundle_used_fallback_scores": False,
                "run_dir": str(run_dir),
                "summary_dir": str(summary_out_dir),
                "run_overview_pdf": str(run_overview_pdf),
            }
            (raw_packet_dir / "packet_meta.json").write_text(json.dumps(packet_meta, indent=2), encoding="utf-8")

            queue_rows.append(
                {
                    "doe_root": str(root),
                    "bucket": bucket_name,
                    "rank_in_bucket": i,
                    "design_id": did,
                    "score_total": _safe_float(drow.get("score_total")),
                    "pass_rate": _safe_float(drow.get("pass_rate")),
                    "rule_name": str(rep.get("rule_name", "")),
                    "seed": seed,
                    "representative_run_dir": str(run_dir),
                    "bundle_packet_dir": str(packet_dir),
                    "bundle_raw_packet_dir": str(raw_packet_dir),
                    "bundle_overview_pdf": str(run_overview_pdf),
                    "bundle_used_fallback_scores": False,
                }
            )
            packet_rows.append(packet_meta)

    queue_csv = bundle_root / "doe_review_queue.csv"
    pd.DataFrame(queue_rows, columns=queue_cols).to_csv(queue_csv, index=False)
    bucket_manifest_csv = raw_dir / "bundle_packets.csv"
    pd.DataFrame(packet_rows).to_csv(bucket_manifest_csv, index=False)

    corr_csv, effects_csv, analysis_summary_pdf_raw = _build_knob_metric_artifacts(
        points_df=design_points,
        design_scores_df=design_scores,
        run_features_df=run_features,
        knob_cols=knob_cols,
        knob_ranges=knob_ranges,
        out_dir=raw_dir,
        rule_name=str(rule_name),
    )
    analysis_summary_pdf = bundle_root / "doe_analysis_summary.pdf"
    if analysis_summary_pdf_raw.exists():
        shutil.move(str(analysis_summary_pdf_raw), str(analysis_summary_pdf))

    return DOEReviewBundleArtifacts(
        bundle_root=bundle_root,
        queue_csv=queue_csv,
        bucket_manifest_csv=bucket_manifest_csv,
        knob_ranges_csv=knob_ranges_csv,
        knob_correlations_csv=corr_csv,
        knob_effects_csv=effects_csv,
        analysis_summary_pdf=analysis_summary_pdf,
    )
