from __future__ import annotations

from dataclasses import dataclass
from math import factorial, log
from pathlib import Path
from typing import Any
import json
import hashlib
import re

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, BoundaryNorm

from src.utils.metrics import gini_index_0_100
from src.viz.color_palette import COLORS as SIM_COLORS
from src.viz.group_palette import get_group_color
from src.analysis.reference_benchmarks import (
    l1_dist,
    utilitarian_ref_l2sq,
    nash_ref_kl,
    rawlsian_ref_minimax_l2sq,
    egalitarian_refs_mean_plus_lambda_gini,
)


@dataclass(frozen=True)
class RunSummaryArtifacts:
    out_dir: Path
    global_series_csv: Path
    area_series_csv: Path
    summary_stats_json: Path
    static_overview_pdf: Path | None = None
    global_summary_pdf: Path | None = None


SUMMARY_MODE_FULL = "full"
SUMMARY_MODE_FAST = "fast"
_SUMMARY_MODES = {SUMMARY_MODE_FULL, SUMMARY_MODE_FAST}


def list_summary_pdfs_in_recommended_view_order(out_dir: Path) -> list[Path]:
    """Return summary PDFs in the order a user should read them.

    Current order contract:
    1) static_overview.pdf
    2) global_summary.pdf
    3) areas_overview.pdf
    4) area_<id>.pdf (ascending area id)
    """
    out = Path(out_dir)
    ordered: list[Path] = []

    fixed = ("static_overview.pdf", "global_summary.pdf", "areas_overview.pdf")
    for name in fixed:
        p = out / name
        if p.exists():
            ordered.append(p)

    area_files: list[tuple[int, Path]] = []
    for p in out.glob("area_*.pdf"):
        m = re.fullmatch(r"area_(\d+)\.pdf", p.name)
        if m is None:
            continue
        area_files.append((int(m.group(1)), p))
    area_files.sort(key=lambda x: x[0])
    ordered.extend([p for _, p in area_files])
    return ordered


def generate_run_summary_batch1(
    run_dir: Path,
    out_dir: Path | None = None,
    *,
    mode: str = SUMMARY_MODE_FULL,
    use_cache: bool = True,
) -> RunSummaryArtifacts:
    """Generate core per-run summary sidecars from schema-v2 logged artifacts only.

    Batch-1 scope:
    - `summary_global_series.csv`
    - `summary_area_series.csv`
    - `summary_stats.json`
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir is not None else (run_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    _validate_summary_mode(mode)

    steps = pd.read_parquet(run_dir / "steps.parquet").sort_values("step").reset_index(drop=True)
    area_steps = pd.read_parquet(run_dir / "area_steps.parquet").sort_values(["step", "area_id"]).reset_index(drop=True)
    agents = pd.read_parquet(run_dir / "agents.parquet").sort_values(["step", "agent_id"]).reset_index(drop=True)
    votes = pd.read_parquet(run_dir / "votes.parquet").sort_values(["step", "area_id", "agent_id"]).reset_index(drop=True)
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
    static = json.loads((run_dir / "static.json").read_text(encoding="utf-8"))

    num_colors = int(static.get("num_colors", 0))
    if num_colors <= 0:
        raise RuntimeError(f"Invalid num_colors in static.json: {num_colors}")

    area_agent_ids = _load_area_agent_ids_from_static_overlays(run_dir=run_dir)
    refs = _load_or_compute_reference_payload(
        out_dir=out_dir,
        static=static,
        num_colors=num_colors,
        area_agent_ids=area_agent_ids,
        mode=mode,
        use_cache=use_cache,
    )

    global_series = _build_global_series(
        steps=steps,
        area_steps=area_steps,
        agents=agents,
        votes=votes,
        num_colors=num_colors,
        refs_global=refs["global"],
    )
    area_series = _build_area_series(
        area_steps=area_steps,
        votes=votes,
        num_colors=num_colors,
        refs_by_area=refs["areas"],
    )
    stats = _build_summary_stats(
        global_series=global_series,
        area_series=area_series,
        meta=meta,
        static=static,
    )

    global_path = out_dir / "summary_global_series.csv"
    area_path = out_dir / "summary_area_series.csv"
    stats_path = out_dir / "summary_stats.json"
    global_series.to_csv(global_path, index=False)
    area_series.to_csv(area_path, index=False)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return RunSummaryArtifacts(
        out_dir=out_dir,
        global_series_csv=global_path,
        area_series_csv=area_path,
        summary_stats_json=stats_path,
    )


def generate_run_summary_batch2(
    run_dir: Path,
    out_dir: Path | None = None,
    *,
    mode: str = SUMMARY_MODE_FULL,
    use_cache: bool = True,
) -> RunSummaryArtifacts:
    """Generate batch-1 sidecars plus batch-2 run-level PDFs.

    Batch-2 scope:
    - `static_overview.pdf`
    - `global_summary.pdf`
    """
    base = generate_run_summary_batch1(run_dir=run_dir, out_dir=out_dir, mode=mode, use_cache=use_cache)
    run_dir = Path(run_dir)
    _validate_summary_mode(mode)

    global_series = pd.read_csv(base.global_series_csv).sort_values("step").reset_index(drop=True)
    steps = pd.read_parquet(run_dir / "steps.parquet").sort_values("step").reset_index(drop=True)
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
    static = json.loads((run_dir / "static.json").read_text(encoding="utf-8"))
    area_agent_ids = _load_area_agent_ids_from_static_overlays(run_dir=run_dir)
    refs_global = _load_or_compute_reference_payload(
        out_dir=base.out_dir,
        static=static,
        num_colors=int(static.get("num_colors", 0)),
        area_agent_ids=area_agent_ids,
        mode=mode,
        use_cache=use_cache,
    )["global"]

    static_pdf = base.out_dir / "static_overview.pdf"
    global_pdf = base.out_dir / "global_summary.pdf"
    _render_static_overview_pdf(
        out_pdf=static_pdf,
        static=static,
        meta=meta,
    )
    _render_global_summary_pdf(
        out_pdf=global_pdf,
        run_dir=run_dir,
        global_series=global_series,
        steps=steps,
        static=static,
        meta=meta,
        refs_global=refs_global,
    )

    return RunSummaryArtifacts(
        out_dir=base.out_dir,
        global_series_csv=base.global_series_csv,
        area_series_csv=base.area_series_csv,
        summary_stats_json=base.summary_stats_json,
        static_overview_pdf=static_pdf,
        global_summary_pdf=global_pdf,
    )


def _build_global_series(
    *,
    steps: pd.DataFrame,
    area_steps: pd.DataFrame,
    agents: pd.DataFrame,
    votes: pd.DataFrame,
    num_colors: int,
    refs_global: dict[str, np.ndarray | None],
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
    num_colors: int,
    refs_by_area: dict[int, dict[str, np.ndarray | None]],
) -> pd.DataFrame:
    area_color_cols = [f"area_color_{i}" for i in range(num_colors)]
    missing = [c for c in area_color_cols if c not in area_steps.columns]
    if missing:
        raise RuntimeError(f"area_steps.parquet missing required area_color columns: {missing}")

    a = pd.DataFrame(
        {
            "step": area_steps["step"].astype(np.int32),
            "area_id": area_steps["area_id"].astype(np.int32),
            "participants": area_steps["participants"].astype(np.int32),
            "eligible_voters": area_steps["eligible_voters"].astype(np.int32),
            "turnout": area_steps["turnout"].astype(np.float32),
            "gini_assets": area_steps["gini_index"].astype(np.float32),
            "dist_to_reality": area_steps["dist_to_reality"].astype(np.float32),
        }
    )
    for c in area_color_cols:
        a[c] = area_steps[c].astype(np.float32)

    diversity_rows = _diversity_entropy_by_step_area(votes=votes, num_options=factorial(num_colors))
    if diversity_rows.empty:
        a["diversity_first_choice_entropy"] = np.float32(np.nan)
    else:
        a = a.merge(diversity_rows, on=["step", "area_id"], how="left")
        a["diversity_first_choice_entropy"] = a["diversity_first_choice_entropy"].astype(np.float32)

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


def _validate_summary_mode(mode: str) -> None:
    if mode not in _SUMMARY_MODES:
        allowed = ", ".join(sorted(_SUMMARY_MODES))
        raise ValueError(f"Invalid summary mode '{mode}'. Allowed: {allowed}")


def _ref_cache_path(out_dir: Path, mode: str) -> Path:
    return out_dir / f"reference_cache_{mode}.json"


def _reference_cache_key(
    *,
    static: dict[str, Any],
    num_colors: int,
    area_agent_ids: dict[int, list[int]],
    mode: str,
) -> str:
    payload = {
        "version": 1,
        "mode": mode,
        "num_colors": int(num_colors),
        "personal_opt_dist": static.get("personal_opt_dist", {}),
        "area_agent_ids": {str(int(k)): [int(v) for v in vals] for k, vals in sorted(area_agent_ids.items())},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _serialize_refs(refs: dict[str, np.ndarray | None]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in refs.items():
        out[k] = None if v is None else np.asarray(v, dtype=np.float64).tolist()
    return out


def _deserialize_refs(raw: dict[str, Any]) -> dict[str, np.ndarray | None]:
    out: dict[str, np.ndarray | None] = {}
    for k, v in raw.items():
        if v is None:
            out[k] = None
        else:
            arr = np.asarray(v, dtype=np.float64)
            out[k] = arr if arr.ndim == 1 else None
    return out


def _compute_reference_set_for_dists(
    *,
    dists: np.ndarray,
    mode: str,
) -> dict[str, np.ndarray | None]:
    if dists.size == 0:
        return {
            "dist_to_ref_utilitarian": None,
            "dist_to_ref_nash": None,
            "dist_to_ref_rawlsian": None,
            "dist_to_ref_egalitarian": None,
            "dist_to_ref_egalitarian_lam025": None,
            "dist_to_ref_egalitarian_lam400": None,
        }

    util = utilitarian_ref_l2sq(dists)
    nash = nash_ref_kl(dists)
    if mode == SUMMARY_MODE_FAST:
        return {
            "dist_to_ref_utilitarian": np.asarray(util, dtype=np.float64),
            "dist_to_ref_nash": np.asarray(nash, dtype=np.float64),
            "dist_to_ref_rawlsian": None,
            "dist_to_ref_egalitarian": None,
            "dist_to_ref_egalitarian_lam025": None,
            "dist_to_ref_egalitarian_lam400": None,
        }

    rawl = rawlsian_ref_minimax_l2sq(dists)
    egal_refs = egalitarian_refs_mean_plus_lambda_gini(dists)
    return {
        "dist_to_ref_utilitarian": np.asarray(util, dtype=np.float64),
        "dist_to_ref_nash": np.asarray(nash, dtype=np.float64),
        "dist_to_ref_rawlsian": np.asarray(rawl, dtype=np.float64),
        "dist_to_ref_egalitarian": np.asarray(egal_refs.lam_mid, dtype=np.float64),
        "dist_to_ref_egalitarian_lam025": np.asarray(egal_refs.lam_low, dtype=np.float64),
        "dist_to_ref_egalitarian_lam400": np.asarray(egal_refs.lam_high, dtype=np.float64),
    }


def _load_or_compute_reference_payload(
    *,
    out_dir: Path,
    static: dict[str, Any],
    num_colors: int,
    area_agent_ids: dict[int, list[int]],
    mode: str,
    use_cache: bool,
) -> dict[str, Any]:
    _validate_summary_mode(mode)
    key = _reference_cache_key(
        static=static,
        num_colors=num_colors,
        area_agent_ids=area_agent_ids,
        mode=mode,
    )
    cache_path = _ref_cache_path(out_dir=out_dir, mode=mode)

    if use_cache and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                isinstance(cached, dict)
                and cached.get("cache_key") == key
                and isinstance(cached.get("global"), dict)
                and isinstance(cached.get("areas"), dict)
            ):
                return {
                    "global": _deserialize_refs(cached["global"]),
                    "areas": {
                        int(k): _deserialize_refs(v)
                        for k, v in cached["areas"].items()
                        if isinstance(v, dict)
                    },
                }
        except Exception:
            pass

    d_global = _personal_dists_from_static(static=static, num_colors=num_colors)
    refs_global = _compute_reference_set_for_dists(dists=d_global, mode=mode)

    personal_by_id = static.get("personal_opt_dist", {}) if isinstance(static.get("personal_opt_dist"), dict) else {}
    refs_by_area: dict[int, dict[str, np.ndarray | None]] = {}
    for area_id, ids in area_agent_ids.items():
        rows: list[np.ndarray] = []
        for aid in ids:
            arr = np.asarray(personal_by_id.get(str(int(aid))), dtype=np.float64)
            if arr.ndim == 1 and arr.size == int(num_colors):
                s = float(np.sum(arr))
                if s > 0.0:
                    rows.append(arr / s)
        d_area = np.vstack(rows).astype(np.float64) if rows else np.asarray([], dtype=np.float64)
        refs_by_area[int(area_id)] = _compute_reference_set_for_dists(dists=d_area, mode=mode)

    payload = {
        "cache_key": key,
        "global": _serialize_refs(refs_global),
        "areas": {str(int(k)): _serialize_refs(v) for k, v in refs_by_area.items()},
    }
    if use_cache:
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"global": refs_global, "areas": refs_by_area}


def _weighted_dist_to_reality_by_step(*, area_steps: pd.DataFrame, step_index: np.ndarray) -> list[float]:
    out: list[float] = []
    grouped = area_steps.groupby("step", sort=True)[["dist_to_reality", "eligible_voters"]]
    for step in step_index:
        if int(step) not in grouped.groups:
            out.append(float("nan"))
            continue
        block = grouped.get_group(int(step))
        weights = block["eligible_voters"].to_numpy(dtype=float)
        vals = block["dist_to_reality"].to_numpy(dtype=float)
        denom = float(np.sum(weights))
        if denom <= 0.0:
            out.append(float("nan"))
        else:
            out.append(float(np.sum(vals * weights) / denom))
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


def _build_summary_stats(
    *,
    global_series: pd.DataFrame,
    area_series: pd.DataFrame,
    meta: dict[str, Any],
    static: dict[str, Any],
) -> dict[str, Any]:
    def _safe_mean(series: pd.Series) -> float:
        arr = series.to_numpy(dtype=float)
        finite = arr[np.isfinite(arr)]
        return float(np.mean(finite)) if finite.size > 0 else float("nan")

    def _safe_final(series: pd.Series) -> float:
        arr = series.to_numpy(dtype=float)
        if arr.size == 0:
            return float("nan")
        return float(arr[-1])

    summary = {
        "run": {
            "run_seed": int(meta["run"]["run_seed"]),
            "rule_idx": int(meta["run"]["rule_idx"]),
            "rule_name": meta["run"].get("rule_name"),
            "distance_name": meta["run"].get("distance_name"),
        },
        "shape": {
            "num_steps": int(global_series["step"].nunique()),
            "num_areas": int(area_series["area_id"].nunique()),
            "num_agents": int(static.get("num_agents", 0)),
            "num_colors": int(static.get("num_colors", 0)),
        },
        "global_summary": {
            "turnout_mean": _safe_mean(global_series["turnout"]),
            "turnout_final": _safe_final(global_series["turnout"]),
            "gini_assets_mean": _safe_mean(global_series["gini_assets"]),
            "gini_assets_final": _safe_final(global_series["gini_assets"]),
            "gini_dissatisfaction_mean": _safe_mean(global_series["gini_dissatisfaction"]),
            "gini_dissatisfaction_final": _safe_final(global_series["gini_dissatisfaction"]),
            "mean_dissatisfaction_mean": _safe_mean(global_series["mean_dissatisfaction"]),
            "mean_dissatisfaction_final": _safe_final(global_series["mean_dissatisfaction"]),
            "dist_to_reality_mean": _safe_mean(global_series["dist_to_reality"]),
            "dist_to_reality_final": _safe_final(global_series["dist_to_reality"]),
            "diversity_entropy_mean": _safe_mean(global_series["diversity_first_choice_entropy"]),
            "diversity_entropy_final": _safe_final(global_series["diversity_first_choice_entropy"]),
        },
    }
    return summary


def _render_static_overview_pdf(*, out_pdf: Path, static: dict[str, Any], meta: dict[str, Any]) -> None:
    num_colors = int(static.get("num_colors", 0))
    num_areas = int(static.get("num_areas", 0))
    num_agents = int(static.get("num_agents", 0))
    width = int(static.get("width", 0))
    height = int(static.get("height", 0))
    run_seed = int(meta["run"]["run_seed"])
    rule_name = meta["run"].get("rule_name")
    distance_name = meta["run"].get("distance_name")

    info = static.get("personality_group_info", {}) or {}
    personality_groups = np.asarray(info.get("personality_groups", []), dtype=int)
    global_dist = np.asarray(info.get("global_distribution", []), dtype=float)
    areas_info = info.get("areas", {}) or {}

    if personality_groups.ndim != 2:
        personality_groups = np.zeros((0, num_colors), dtype=int)
    n_groups = int(personality_groups.shape[0])

    with PdfPages(out_pdf) as pdf:
        # Single-page layout:
        # left-top: personality order with color blocks
        # left-bottom: global group distribution
        # right: per-area group composition (full height)
        fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
        outer = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.6])
        left = outer[0, 0].subgridspec(2, 1, height_ratios=[1.2, 0.8])

        ax_map = fig.add_subplot(left[0, 0])
        ax_global = fig.add_subplot(left[1, 0])
        ax_area = fig.add_subplot(outer[0, 1])

        fig.suptitle(
            f"Static Overview | run_seed={run_seed} | rule={rule_name} | distance={distance_name} | "
            f"grid={width}x{height} | agents={num_agents} | areas={num_areas} | colors={num_colors}",
            fontsize=11,
        )

        _draw_personality_group_order_block(
            ax=ax_map,
            personality_groups=personality_groups,
            num_colors=num_colors,
        )

        # Global group distribution
        if n_groups > 0 and global_dist.size == n_groups:
            x = np.arange(n_groups)
            bar_colors = [get_group_color(gi) for gi in range(n_groups)]
            ax_global.bar(x, global_dist, color=bar_colors, alpha=0.9)
            ax_global.set_xticks(x)
            ax_global.set_xticklabels([f"g{i}" for i in range(n_groups)])
            ax_global.set_ylim(0.0, 1.0)
            if n_groups > 0:
                major = int(np.argmax(global_dist))
                ax_global.text(
                    0.99,
                    0.98,
                    f"majority: g{major} ({100.0 * float(global_dist[major]):.1f}%)",
                    transform=ax_global.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                )
        else:
            ax_global.text(0.5, 0.5, "No global group metadata", ha="center", va="center")
        ax_global.set_title("Global Personality Group Distribution")
        ax_global.set_ylabel("share")
        ax_global.grid(True, axis="y", alpha=0.25)

        # Per-area group composition
        area_rows: list[tuple[str, int, np.ndarray]] = []
        for area_key in sorted(areas_info.keys(), key=lambda x: int(x)):
            payload = areas_info.get(area_key) or {}
            area_n = int(payload.get("num_agents", 0))
            dist = np.asarray(payload.get("personality_group_distribution", []), dtype=float)
            area_rows.append((area_key, area_n, dist))

        if area_rows and n_groups > 0:
            n_area = len(area_rows)
            if n_area < 16:
                # Keep visual density comparable to larger-area runs:
                # center rows inside a virtual 16-row frame.
                y_offset = 0.5 * (16 - n_area)
                y = np.arange(n_area, dtype=float) + y_offset
                bar_h = 0.55
                ax_area.set_ylim(-0.5, 15.5)
            else:
                y = np.arange(n_area, dtype=float)
                bar_h = 0.8
            left_vals = np.zeros(len(area_rows), dtype=float)
            for gi in range(n_groups):
                vals = np.array(
                    [float(r[2][gi]) if r[2].size > gi else 0.0 for r in area_rows],
                    dtype=float,
                )
                ax_area.barh(
                    y,
                    vals,
                    left=left_vals,
                    height=bar_h,
                    label=f"g{gi}",
                    color=get_group_color(gi),
                )
                left_vals += vals
            labels = [f"a{a} (n={n})" for a, n, _ in area_rows]
            ax_area.set_yticks(y)
            ax_area.set_yticklabels(labels)
            ax_area.set_xlim(0.0, 1.0)
            ax_area.legend(loc="lower right", fontsize=8, ncol=2)
        else:
            ax_area.text(0.5, 0.5, "No area group metadata", ha="center", va="center")
        ax_area.set_title("Per-Area Group Composition")
        ax_area.set_xlabel("share")
        ax_area.grid(True, axis="x", alpha=0.25)

        fig.tight_layout()
        pdf.savefig(fig, dpi=140)
        plt.close(fig)


def _render_global_summary_pdf(
    *,
    out_pdf: Path,
    run_dir: Path,
    global_series: pd.DataFrame,
    steps: pd.DataFrame,
    static: dict[str, Any],
    meta: dict[str, Any],
    refs_global: dict[str, np.ndarray | None],
) -> None:
    with PdfPages(out_pdf) as pdf:
        _render_global_core_metrics_page(pdf=pdf, global_series=global_series, meta=meta)
        _render_global_distance_page(pdf=pdf, global_series=global_series)
        _render_global_colors_and_grids_page(
            pdf=pdf,
            run_dir=run_dir,
            global_series=global_series,
            steps=steps,
            static=static,
            refs_global=refs_global,
        )


def _render_global_core_metrics_page(*, pdf: PdfPages, global_series: pd.DataFrame, meta: dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), sharex=True)
    x = global_series["step"].to_numpy(dtype=float)
    ax = axes.ravel()
    ax[0].plot(x, global_series["turnout"].to_numpy(dtype=float), color="tab:blue")
    ax[0].set_title("Turnout [%]")
    ax[1].plot(x, global_series["gini_assets"].to_numpy(dtype=float), color="tab:red")
    ax[1].set_title("Gini Assets [0..100]")
    ax[2].plot(x, global_series["gini_dissatisfaction"].to_numpy(dtype=float), color="tab:purple")
    ax[2].set_title("Gini Dissatisfaction [0..100]")
    ax[3].plot(x, global_series["mean_dissatisfaction"].to_numpy(dtype=float), color="tab:orange")
    ax[3].set_title("Mean Dissatisfaction")
    for a in ax:
        a.grid(True, alpha=0.25)
        a.set_xlabel("step")
    title = f"Global Core Metrics | run_seed={meta['run']['run_seed']} | rule={meta['run'].get('rule_name')}"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    pdf.savefig(fig, dpi=140)
    plt.close(fig)


def _render_global_distance_page(*, pdf: PdfPages, global_series: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), sharex=True)
    x = global_series["step"].to_numpy(dtype=float)
    ax = axes.ravel()
    ax[0].plot(x, global_series["dist_to_reality"].to_numpy(dtype=float), color="tab:green")
    ax[0].set_title("dist_to_reality (weighted)")
    ax[1].plot(x, global_series["dist_to_ref_utilitarian"].to_numpy(dtype=float), color="tab:blue", label="utilitarian")
    if "dist_to_ref_nash" in global_series.columns:
        ax[1].plot(x, global_series["dist_to_ref_nash"].to_numpy(dtype=float), color="tab:purple", label="nash")
    ax[1].plot(x, global_series["dist_to_ref_egalitarian"].to_numpy(dtype=float), color="tab:orange", label="egalitarian")
    ax[1].plot(x, global_series["dist_to_ref_rawlsian"].to_numpy(dtype=float), color="tab:red", label="rawlsian")
    ax[1].set_title("dist_to_ref_*")
    ax[1].legend(loc="best", fontsize=8)
    ax[2].plot(x, global_series["diversity_first_choice_entropy"].to_numpy(dtype=float), color="tab:brown")
    ax[2].set_title("diversity_first_choice_entropy")
    ax[3].axis("off")
    ax[3].text(
        0.02,
        0.98,
        "Distance metrics are lower-better.\nDiversity entropy is normalized to [0,1].\nNaN means no participants for that step.",
        va="top",
        ha="left",
        fontsize=10,
    )
    for a in ax[:3]:
        a.grid(True, alpha=0.25)
        a.set_xlabel("step")
    fig.suptitle("Global Distance + Diversity Diagnostics", fontsize=12)
    fig.tight_layout()
    pdf.savefig(fig, dpi=140)
    plt.close(fig)


def _render_global_colors_and_grids_page(
    *,
    pdf: PdfPages,
    run_dir: Path,
    global_series: pd.DataFrame,
    steps: pd.DataFrame,
    static: dict[str, Any],
    refs_global: dict[str, np.ndarray | None],
) -> None:
    num_colors = int(static.get("num_colors", 0))
    color_cols = [f"color_{i}" for i in range(num_colors) if f"color_{i}" in global_series.columns]

    refs = {
        "utilitarian": refs_global.get("dist_to_ref_utilitarian"),
        "nash": refs_global.get("dist_to_ref_nash"),
        "egalitarian": refs_global.get("dist_to_ref_egalitarian"),
        "rawlsian": refs_global.get("dist_to_ref_rawlsian"),
    }

    fig = plt.figure(figsize=(11.69, 8.27))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0], width_ratios=[1.1, 1.4, 1.4])
    ax_ref = fig.add_subplot(gs[0, 0])
    ax_curve = fig.add_subplot(gs[0, 1:])
    x = global_series["step"].to_numpy(dtype=float)
    for i, c in enumerate(color_cols):
        ax_curve.plot(
            x,
            global_series[c].to_numpy(dtype=float),
            label=f"color_{i}",
            color=_sim_color(i),
        )
    ax_curve.set_title("Global Color Distribution Curves")
    ax_curve.set_xlabel("step")
    ax_curve.set_ylabel("share")
    ax_curve.set_ylim(0.0, 1.0)
    ax_curve.grid(True, alpha=0.25)
    if color_cols:
        ax_curve.legend(loc="upper right", ncol=min(5, len(color_cols)), fontsize=8)
    _draw_reference_optima_panel(ax=ax_ref, refs=refs, num_colors=num_colors)

    # Grid snapshots: step 1 and last step (carry-forward if sparse interval).
    step_last = int(steps["step"].max()) if len(steps) > 0 else 1
    grid1 = _load_grid_with_carry_forward(run_dir=run_dir, step=1, max_step=step_last)
    grid_last = _load_grid_with_carry_forward(run_dir=run_dir, step=step_last, max_step=step_last)
    ax_note = fig.add_subplot(gs[1, 0])
    ax_g1 = fig.add_subplot(gs[1, 1])
    ax_gn = fig.add_subplot(gs[1, 2])
    ax_note.axis("off")
    ax_note.text(
        0.02,
        0.98,
        "Reference panel (top-left):\n"
        "four fixed benchmark distributions\n"
        "shown as colored dots per reference\n"
        "(y-axis = share 0..1),\n"
        "used by dist_to_ref_*.\n\n"
        "Color IDs and hues match simulation colors,\n"
        "so composition can be compared directly\n"
        "to time-varying global color curves.",
        va="top",
        ha="left",
        fontsize=9,
    )
    _draw_grid_or_note(ax=ax_g1, grid=grid1, title="Grid Snapshot @ step 1")
    _draw_grid_or_note(ax=ax_gn, grid=grid_last, title=f"Grid Snapshot @ step {step_last}")

    fig.tight_layout()
    pdf.savefig(fig, dpi=140)
    plt.close(fig)


def _load_grid_with_carry_forward(*, run_dir: Path, step: int, max_step: int) -> np.ndarray | None:
    grids_dir = run_dir / "grids"
    if not grids_dir.exists():
        return None
    pad = len(str(int(max_step)))
    target = grids_dir / f"grid_{int(step):0{pad}d}.npy"
    if target.exists():
        return np.asarray(np.load(target))
    # Carry-forward: latest available <= step
    candidates: list[int] = []
    for p in grids_dir.glob("grid_*.npy"):
        stem = p.stem
        raw = stem.replace("grid_", "")
        try:
            idx = int(raw)
        except ValueError:
            continue
        if idx <= int(step):
            candidates.append(idx)
    if not candidates:
        return None
    chosen = max(candidates)
    chosen_path = grids_dir / f"grid_{int(chosen):0{pad}d}.npy"
    if not chosen_path.exists():
        return None
    return np.asarray(np.load(chosen_path))


def _draw_grid_or_note(*, ax, grid: np.ndarray | None, title: str) -> None:
    ax.set_title(title)
    if grid is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "Grid snapshot not available", ha="center", va="center")
        return
    palette = [_sim_color(i) for i in range(int(np.nanmax(grid)) + 1)]
    cmap = ListedColormap(palette)
    bounds = np.arange(-0.5, len(palette) + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    ax.imshow(grid, interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])


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


def _load_area_agent_ids_from_static_overlays(*, run_dir: Path) -> dict[int, list[int]]:
    """Recover resident area->agent ids from static overlay arrays written at run start."""
    area_map: dict[int, set[int]] = {}
    area_path = run_dir / "area_strings_per_cell.npy"
    agent_path = run_dir / "agent_strings_per_cell.npy"
    if not area_path.exists() or not agent_path.exists():
        return {}

    area_grid = np.load(area_path, allow_pickle=True)
    agent_grid = np.load(agent_path, allow_pickle=True)
    if area_grid.shape != agent_grid.shape:
        return {}

    for idx in np.ndindex(area_grid.shape):
        area_str = str(area_grid[idx]) if area_grid[idx] is not None else ""
        agent_str = str(agent_grid[idx]) if agent_grid[idx] is not None else ""
        if not area_str.strip() or not agent_str.strip():
            continue

        area_ids: list[int] = []
        for tok in area_str.split(","):
            tok = tok.strip()
            if tok == "":
                continue
            try:
                area_ids.append(int(tok))
            except ValueError:
                continue
        if not area_ids:
            continue

        # agent_strings are formatted as "<id>: <...>, <id>: <...>"
        agent_ids = [int(m.group(1)) for m in re.finditer(r"(\d+)\s*:", agent_str)]
        if not agent_ids:
            continue

        for a_id in area_ids:
            if a_id < 0:
                continue
            if a_id not in area_map:
                area_map[a_id] = set()
            area_map[a_id].update(agent_ids)

    return {k: sorted(v) for k, v in area_map.items()}


def _draw_reference_optima_panel(*, ax, refs: dict[str, np.ndarray | None], num_colors: int) -> None:
    ax.set_title("Fixed Reference Optima")
    names = ("utilitarian", "nash", "egalitarian", "rawlsian")
    x_pos = np.arange(len(names), dtype=float)
    any_valid = False

    for xi, name in enumerate(names):
        ref = refs.get(name)
        if ref is None or ref.size != int(num_colors):
            continue
        for color_id in range(int(num_colors)):
            x_val = float(x_pos[xi])
            y_val = float(ref[color_id])
            # short horizontal segment centered on dot to separate close values
            ax.plot(
                [x_val - 0.185, x_val + 0.185],
                [y_val, y_val],
                color=_sim_color(color_id),
                linewidth=1.0,
                alpha=0.9,
                zorder=2,
            )
            ax.scatter(
                [x_val],
                [y_val],
                s=34,
                color=_sim_color(color_id),
                edgecolors="black",
                linewidths=0.35,
                zorder=3,
            )
            any_valid = True

    if not any_valid:
        ax.text(0.5, 0.5, "reference distributions unavailable", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    ax.set_xlim(-0.5, len(names) - 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["util", "nash", "egal", "rawl"], fontsize=8)
    ax.set_ylabel("share")
    ax.grid(True, axis="y", alpha=0.25)

def _sim_color(color_idx: int) -> str:
    if 0 <= int(color_idx) < len(SIM_COLORS):
        name = SIM_COLORS[int(color_idx)]
        # Matplotlib normalizes both spellings; keep a single one for consistency.
        return "LightGrey" if name == "LightGray" else str(name)
    return "black"


def _draw_personality_group_order_block(*, ax, personality_groups: np.ndarray, num_colors: int) -> None:
    ax.set_title("Personality Group -> Color Preference Order")
    if personality_groups.ndim != 2 or personality_groups.shape[0] == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "No personality group metadata", ha="center", va="center")
        return

    n_groups = int(personality_groups.shape[0])
    n_slots = int(min(num_colors, personality_groups.shape[1]))
    ax.set_xlim(0.0, float(n_slots + 1.8))
    ax.set_ylim(-0.5, float(n_groups - 0.5))
    ax.invert_yaxis()
    ax.set_yticks(np.arange(n_groups))
    ax.set_yticklabels([f"g{i}" for i in range(n_groups)])
    ax.set_xticks([])
    ax.grid(False)

    for gi in range(n_groups):
        order = personality_groups[gi].astype(int).tolist()
        for pos, color_id in enumerate(order[:n_slots]):
            x0 = float(pos + 1.0)
            y0 = float(gi - 0.32)
            rect = plt.Rectangle(
                (x0, y0),
                0.9,
                0.64,
                facecolor=_sim_color(color_id),
                edgecolor="black",
                linewidth=0.6,
            )
            ax.add_patch(rect)
            ax.text(
                x0 + 0.45,
                y0 - 0.06,
                str(pos + 1),
                ha="center",
                va="bottom",
                fontsize=7,
                color="black",
            )
            ax.text(
                x0 + 0.45,
                y0 + 0.32,
                str(int(color_id)),
                ha="center",
                va="center",
                fontsize=7,
                color="black",
                fontweight="bold",
            )

    for s in ax.spines.values():
        s.set_alpha(0.3)
