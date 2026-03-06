from __future__ import annotations

from dataclasses import dataclass
from math import factorial, log
from pathlib import Path
from typing import Any
import json
import hashlib
import re
import itertools
import warnings

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

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
from src.analysis.thesis_endpoints import (
    step_volatility_l1_normalized,
    time_mean,
)
from src.analysis.doe_scoring import DEFAULT_SCORING_THRESHOLDS
from src.utils.ballots import score_options_c2
from src.utils.distance_functions import spearman_fr_order, kendall_tau_order
from src.utils.representations import distribution_to_ordering_tie_aware
from src.utils.social_welfare_functions import majority_rule, approval_voting, utilitarian_rule, borda_rule, random_rule


@dataclass(frozen=True)
class RunSummaryArtifacts:
    out_dir: Path
    global_series_csv: Path
    area_series_csv: Path
    summary_stats_json: Path
    area_group_series_csv: Path | None = None
    static_overview_pdf: Path | None = None
    global_summary_pdf: Path | None = None


SUMMARY_MODE_FULL = "full"
SUMMARY_MODE_FAST = "fast"
_SUMMARY_MODES = {SUMMARY_MODE_FULL, SUMMARY_MODE_FAST}

SUMMARY_PROFILE_FULL = "full"
SUMMARY_PROFILE_DEBUG_DOE_COMPACT = "debug_doe_compact"
SUMMARY_PROFILE_THESIS_CORE = "thesis_core"
_SUMMARY_PROFILES = {
    SUMMARY_PROFILE_FULL,
    SUMMARY_PROFILE_DEBUG_DOE_COMPACT,
    SUMMARY_PROFILE_THESIS_CORE,
}


@dataclass(frozen=True)
class _SummaryRenderProfile:
    name: str
    global_colors_and_grids: bool
    global_static_overview: bool
    global_per_area_group_distribution: bool
    global_core_metrics: bool
    global_step_volatility_page: bool
    global_distance_metrics: bool
    area_core_page: bool
    area_puzzle_page: bool
    area_vote_mode_alignment_page: bool
    area_group_opportunity_page: bool
    area_puzzle_gate_page: bool
    area_group_diagnostics_pages: bool
    area_learning_causal_page: bool
    area_assets_page: bool
    area_group_means_page: bool
    area_dist_to_ref_page: bool

_Y_PAD_UNIT = 0.02
_Y_PAD_PERCENT = 1.5
_SMOOTH_WINDOW_STEPS = 9
_MODE_ALIGNMENT_LOW_SUPPORT_VOTES = 5
_SMALL_GROUP_MIN_RESIDENTS = 5


def _set_unit_ylim_visible(ax, *, pad: float = _Y_PAD_UNIT) -> None:
    """Bounded [0,1] axis with a tiny pad so flat lines at 0/1 stay visible."""
    ax.set_ylim(-float(pad), 1.0 + float(pad))


def _set_percent_ylim_visible(ax, *, pad: float = _Y_PAD_PERCENT) -> None:
    """Bounded [0,100] axis with a tiny pad so flat lines at 0/100 stay visible."""
    ax.set_ylim(-float(pad), 100.0 + float(pad))


def _rolling_mean_nan(values: np.ndarray, *, window: int = _SMOOTH_WINDOW_STEPS) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    w = max(1, int(window))
    if w <= 1:
        return arr
    return (
        pd.Series(arr, dtype=float)
        .rolling(window=w, min_periods=1, center=True)
        .mean()
        .to_numpy(dtype=float)
    )


def list_summary_pdfs_in_recommended_view_order(out_dir: Path) -> list[Path]:
    """Return summary PDFs in the order a user should read them.

    Current order contract:
    1) global_summary_<rule>_seed<seed>.pdf
    2) areas_overview.pdf
    3) area_<id>.pdf (ascending area id)
    """
    out = Path(out_dir)
    ordered: list[Path] = []

    global_summaries = sorted(out.glob("global_summary_*_seed*.pdf"))
    ordered.extend(global_summaries)
    p = out / "areas_overview.pdf"
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


@dataclass(frozen=True)
class _RunMetaStaticParseResult:
    ok: bool
    meta: dict[str, Any] | None
    static: dict[str, Any] | None
    num_colors: int | None
    error: str | None


def _parse_required_run_meta_static(*, run_dir: Path) -> _RunMetaStaticParseResult:
    meta_path = run_dir / "meta.yaml"
    static_path = run_dir / "static.json"
    missing = [str(p) for p in (meta_path, static_path) if not p.exists()]
    if missing:
        return _RunMetaStaticParseResult(
            ok=False,
            meta=None,
            static=None,
            num_colors=None,
            error=f"Missing required run metadata artifact(s): {', '.join(missing)}",
        )

    try:
        meta_raw = meta_path.read_text(encoding="utf-8")
    except OSError as e:
        return _RunMetaStaticParseResult(
            ok=False,
            meta=None,
            static=None,
            num_colors=None,
            error=f"Failed to read meta.yaml at {meta_path}: {e}",
        )
    try:
        meta_obj = yaml.safe_load(meta_raw)
    except yaml.YAMLError as e:
        return _RunMetaStaticParseResult(
            ok=False,
            meta=None,
            static=None,
            num_colors=None,
            error=f"Failed to parse YAML in {meta_path}: {e}",
        )
    if not isinstance(meta_obj, dict):
        return _RunMetaStaticParseResult(
            ok=False,
            meta=None,
            static=None,
            num_colors=None,
            error=f"Invalid meta.yaml root in {meta_path}: expected mapping",
        )
    run_meta = meta_obj.get("run")
    if not isinstance(run_meta, dict):
        return _RunMetaStaticParseResult(
            ok=False,
            meta=None,
            static=None,
            num_colors=None,
            error=f"Invalid or missing 'run' section in {meta_path}",
        )
    for req_key in ("run_seed", "rule_idx"):
        if req_key not in run_meta:
            return _RunMetaStaticParseResult(
                ok=False,
                meta=None,
                static=None,
                num_colors=None,
                error=f"Missing required meta.run field '{req_key}' in {meta_path}",
            )
    try:
        int(run_meta["run_seed"])
        int(run_meta["rule_idx"])
    except (TypeError, ValueError) as e:
        return _RunMetaStaticParseResult(
            ok=False,
            meta=None,
            static=None,
            num_colors=None,
            error=f"Invalid meta.run seed/rule fields in {meta_path}: {e}",
        )

    try:
        static_raw = static_path.read_text(encoding="utf-8")
    except OSError as e:
        return _RunMetaStaticParseResult(
            ok=False,
            meta=None,
            static=None,
            num_colors=None,
            error=f"Failed to read static.json at {static_path}: {e}",
        )
    try:
        static_obj = json.loads(static_raw)
    except json.JSONDecodeError as e:
        return _RunMetaStaticParseResult(
            ok=False,
            meta=None,
            static=None,
            num_colors=None,
            error=f"Failed to parse JSON in {static_path}: {e}",
        )
    if not isinstance(static_obj, dict):
        return _RunMetaStaticParseResult(
            ok=False,
            meta=None,
            static=None,
            num_colors=None,
            error=f"Invalid static.json root in {static_path}: expected object",
        )

    if "num_colors" not in static_obj:
        return _RunMetaStaticParseResult(
            ok=False,
            meta=None,
            static=None,
            num_colors=None,
            error=f"Missing required field 'num_colors' in {static_path}",
        )
    try:
        num_colors = int(static_obj["num_colors"])
    except (TypeError, ValueError) as e:
        return _RunMetaStaticParseResult(
            ok=False,
            meta=None,
            static=None,
            num_colors=None,
            error=f"Invalid static.json field 'num_colors' in {static_path}: {e}",
        )
    if num_colors <= 0:
        return _RunMetaStaticParseResult(
            ok=False,
            meta=None,
            static=None,
            num_colors=None,
            error=f"Invalid static.json field 'num_colors' in {static_path}: {num_colors}",
        )

    return _RunMetaStaticParseResult(
        ok=True,
        meta=dict(meta_obj),
        static=dict(static_obj),
        num_colors=num_colors,
        error=None,
    )


def _load_required_run_meta_static(*, run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], int]:
    parsed = _parse_required_run_meta_static(run_dir=run_dir)
    if not parsed.ok or parsed.meta is None or parsed.static is None or parsed.num_colors is None:
        raise RuntimeError(parsed.error or f"Failed to parse required run metadata for {run_dir}")
    return parsed.meta, parsed.static, int(parsed.num_colors)


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
    meta, static, num_colors = _load_required_run_meta_static(run_dir=run_dir)

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
        agents=agents,
        area_agent_ids=area_agent_ids,
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
    profile: str = SUMMARY_PROFILE_FULL,
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
    _validate_summary_profile(profile)

    global_series = pd.read_csv(base.global_series_csv).sort_values("step").reset_index(drop=True)
    area_series = pd.read_csv(base.area_series_csv).sort_values(["step", "area_id"]).reset_index(drop=True)
    agents = pd.read_parquet(run_dir / "agents.parquet").sort_values(["step", "agent_id"]).reset_index(drop=True)
    votes = pd.read_parquet(run_dir / "votes.parquet").sort_values(["step", "area_id", "agent_id"]).reset_index(drop=True)
    steps = pd.read_parquet(run_dir / "steps.parquet").sort_values("step").reset_index(drop=True)
    meta, static, num_colors = _load_required_run_meta_static(run_dir=run_dir)
    area_agent_ids = _load_area_agent_ids_from_static_overlays(run_dir=run_dir)
    refs_payload = _load_or_compute_reference_payload(
        out_dir=base.out_dir,
        static=static,
        num_colors=int(num_colors),
        area_agent_ids=area_agent_ids,
        mode=mode,
        use_cache=use_cache,
    )
    refs_global = refs_payload["global"]
    render_profile = _resolve_summary_render_profile(
        profile=profile,
        num_areas=int(area_series["area_id"].nunique()) if "area_id" in area_series.columns else 0,
    )
    area_group_series = _build_area_group_series(
        agents=agents,
        votes=votes,
        area_agent_ids=area_agent_ids,
        participation_alpha=_load_participation_alpha_for_run(run_dir=run_dir),
        participation_signal_mode=_load_participation_signal_mode_for_run(run_dir=run_dir),
        participation_signal_group_shrink_k=_load_participation_signal_group_shrink_k_for_run(run_dir=run_dir),
        altruism_alpha=_load_altruism_alpha_for_run(run_dir=run_dir),
        altruism_learning=_load_altruism_learning_for_run(run_dir=run_dir),
    )
    area_group_path = base.out_dir / "summary_area_group_series.csv"
    area_group_series.to_csv(area_group_path, index=False)

    rule_name = str(meta["run"].get("rule_name", "rule")).strip().lower().replace(" ", "_")
    safe_rule = re.sub(r"[^a-z0-9_\\-]+", "", rule_name) or "rule"
    seed = int(meta["run"]["run_seed"])
    global_pdf = base.out_dir / f"global_summary_{safe_rule}_seed{seed}.pdf"
    _render_combined_global_summary_pdf(
        out_pdf=global_pdf,
        run_dir=run_dir,
        global_series=global_series,
        steps=steps,
        static=static,
        meta=meta,
        refs_global=refs_global,
        render_profile=render_profile,
    )
    _render_area_detail_pdfs(
        run_dir=run_dir,
        out_dir=base.out_dir,
        area_series=area_series,
        area_group_series=area_group_series,
        static=static,
        meta=meta,
        refs_by_area=refs_payload["areas"],
        render_profile=render_profile,
    )

    return RunSummaryArtifacts(
        out_dir=base.out_dir,
        global_series_csv=base.global_series_csv,
        area_series_csv=base.area_series_csv,
        area_group_series_csv=area_group_path,
        summary_stats_json=base.summary_stats_json,
        static_overview_pdf=None,
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
            "non_altruistic_rank1_match_puzzle_share",
            "altruistic_rank1_match_outcome_share",
            "non_altruistic_rank1_match_outcome_share",
            "altruistic_votes_count",
            "non_altruistic_votes_count",
            "vote_count_total",
            "altruistic_vote_share",
            "non_altruistic_vote_share",
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
                "non_altruistic_rank1_match_puzzle_share",
                "altruistic_rank1_match_outcome_share",
                "non_altruistic_rank1_match_outcome_share",
                "altruistic_votes_count",
                "non_altruistic_votes_count",
                "vote_count_total",
                "altruistic_vote_share",
                "non_altruistic_vote_share",
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
        for mode_val, prefix in ((True, "altruistic"), (False, "non_altruistic")):
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
        "non_altruistic_voters",
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
        "non_altruistic_voters_mean_participation_q_delta",
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
        "non_altruistic_voters_mean_participation_q_update_proxy",
        "altruistic_voters_mean_dissatisfaction_signal",
        "non_altruistic_voters_mean_dissatisfaction_signal",
        "altruistic_voters_mean_altruism_delta",
        "non_altruistic_voters_mean_altruism_delta",
        "altruistic_voters_mean_altruism_update_proxy",
        "non_altruistic_voters_mean_altruism_update_proxy",
        "vote_mode_switch_share",
        "vote_mode_switch_from_altruistic_share",
        "vote_mode_switch_from_non_altruistic_share",
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
                "non_altruistic_voters",
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
            v["non_altruistic_voters"] = (v["voted_altruistically"] == False).astype("int32")
        else:
            v["non_altruistic_voters"] = 0
        participants = (
            v.groupby(["step", "area_id", "personality_group_idx"], sort=True, as_index=False)
            .agg(
                participants=("agent_id", "nunique"),
                non_altruistic_voters=("non_altruistic_voters", "sum"),
            )
        )

    out = grouped.merge(
        participants,
        on=["step", "area_id", "personality_group_idx"],
        how="left",
    )
    out["participants"] = out["participants"].fillna(0).astype("int32")
    out["non_altruistic_voters"] = out["non_altruistic_voters"].fillna(0).astype("int32")
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
                        "dissatisfaction_signal": "non_altruistic_voters_mean_dissatisfaction_signal"
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
                        "participation_q_update_proxy": "non_altruistic_voters_mean_participation_q_update_proxy"
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
                        "participation_q_delta": "non_altruistic_voters_mean_participation_q_delta"
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
                .rename(columns={"altruism_delta": "non_altruistic_voters_mean_altruism_delta"})
            )
            v_non_a = (
                v_mode[v_mode["voted_altruistically"] == False]
                .groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)[
                    "altruism_update_proxy"
                ]
                .mean()
                .rename(
                    columns={
                        "altruism_update_proxy": "non_altruistic_voters_mean_altruism_update_proxy"
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
                switch_block["switched_from_non_altruistic"] = (
                    (~switch_block["prev_voted_altruistically"].astype(bool))
                    & switch_block["voted_altruistically"].astype(bool)
                ).astype(int)
                switch_counts = (
                    switch_block.groupby(["step", "area_id", "personality_group_idx"], sort=False, as_index=False)
                    .agg(
                        switched=("switched", "sum"),
                        switched_from_altruistic=("switched_from_altruistic", "sum"),
                        switched_from_non_altruistic=("switched_from_non_altruistic", "sum"),
                    )
                )
                v_switch = current_counts.merge(
                    switch_counts,
                    on=["step", "area_id", "personality_group_idx"],
                    how="left",
                ).fillna({"switched": 0, "switched_from_altruistic": 0, "switched_from_non_altruistic": 0})
                n = v_switch["current_participants"].to_numpy(dtype=float)
                sw = v_switch["switched"].to_numpy(dtype=float)
                sw_a = v_switch["switched_from_altruistic"].to_numpy(dtype=float)
                sw_n = v_switch["switched_from_non_altruistic"].to_numpy(dtype=float)
                v_switch["vote_mode_switch_share"] = np.divide(
                    sw, n, out=np.full_like(sw, np.nan), where=n > 0.0
                )
                v_switch["vote_mode_switch_from_altruistic_share"] = np.divide(
                    sw_a, n, out=np.full_like(sw_a, np.nan), where=n > 0.0
                )
                v_switch["vote_mode_switch_from_non_altruistic_share"] = np.divide(
                    sw_n, n, out=np.full_like(sw_n, np.nan), where=n > 0.0
                )
                v_switch = v_switch[
                    [
                        "step",
                        "area_id",
                        "personality_group_idx",
                        "vote_mode_switch_share",
                        "vote_mode_switch_from_altruistic_share",
                        "vote_mode_switch_from_non_altruistic_share",
                    ]
                ]
            else:
                v_switch = current_counts.copy()
                v_switch["vote_mode_switch_share"] = np.nan
                v_switch["vote_mode_switch_from_altruistic_share"] = np.nan
                v_switch["vote_mode_switch_from_non_altruistic_share"] = np.nan
                v_switch = v_switch[
                    [
                        "step",
                        "area_id",
                        "personality_group_idx",
                        "vote_mode_switch_share",
                        "vote_mode_switch_from_altruistic_share",
                        "vote_mode_switch_from_non_altruistic_share",
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
        "non_altruistic_voters_mean_participation_q_delta",
        "group_mean_participation_p_delta",
        "participants_mean_participation_p_delta",
        "abstainers_mean_participation_p_delta",
        "group_mean_participation_q_update_proxy",
        "participants_mean_participation_q_update_proxy",
        "abstainers_mean_participation_q_update_proxy",
        "altruistic_voters_mean_participation_q_update_proxy",
        "non_altruistic_voters_mean_participation_q_update_proxy",
        "altruistic_voters_mean_dissatisfaction_signal",
        "non_altruistic_voters_mean_dissatisfaction_signal",
        "altruistic_voters_mean_altruism_delta",
        "non_altruistic_voters_mean_altruism_delta",
        "altruistic_voters_mean_altruism_update_proxy",
        "non_altruistic_voters_mean_altruism_update_proxy",
        "vote_mode_switch_share",
        "vote_mode_switch_from_altruistic_share",
        "vote_mode_switch_from_non_altruistic_share",
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


def _validate_summary_mode(mode: str) -> None:
    if mode not in _SUMMARY_MODES:
        allowed = ", ".join(sorted(_SUMMARY_MODES))
        raise ValueError(f"Invalid summary mode '{mode}'. Allowed: {allowed}")


def _validate_summary_profile(profile: str) -> None:
    if profile not in _SUMMARY_PROFILES:
        allowed = ", ".join(sorted(_SUMMARY_PROFILES))
        raise ValueError(f"Invalid summary profile '{profile}'. Allowed: {allowed}")


def _resolve_summary_render_profile(*, profile: str, num_areas: int) -> _SummaryRenderProfile:
    _validate_summary_profile(profile)
    if profile == SUMMARY_PROFILE_FULL:
        return _SummaryRenderProfile(
            name=profile,
            global_colors_and_grids=True,
            global_static_overview=True,
            global_per_area_group_distribution=True,
            global_core_metrics=True,
            global_step_volatility_page=True,
            global_distance_metrics=True,
            area_core_page=True,
            area_puzzle_page=True,
            area_vote_mode_alignment_page=True,
            area_group_opportunity_page=True,
            area_puzzle_gate_page=False,
            area_group_diagnostics_pages=True,
            area_learning_causal_page=True,
            area_assets_page=True,
            area_group_means_page=True,
            area_dist_to_ref_page=True,
        )
    if profile == SUMMARY_PROFILE_DEBUG_DOE_COMPACT:
        # DOE/debug profile keeps area core + group diagnostics enabled for interpretability.
        return _SummaryRenderProfile(
            name=profile,
            global_colors_and_grids=False,
            global_static_overview=False,
            global_per_area_group_distribution=False,
            global_core_metrics=True,
            global_step_volatility_page=True,
            global_distance_metrics=True,
            area_core_page=True,
            area_puzzle_page=True,
            area_vote_mode_alignment_page=True,
            area_group_opportunity_page=True,
            area_puzzle_gate_page=True,
            area_group_diagnostics_pages=True,
            area_learning_causal_page=True,
            area_assets_page=False,
            area_group_means_page=False,
            area_dist_to_ref_page=False,
        )
    # Placeholder profile: keep full behavior until thesis-core packet is implemented.
    return _SummaryRenderProfile(
        name=profile,
        global_colors_and_grids=True,
        global_static_overview=True,
        global_per_area_group_distribution=True,
        global_core_metrics=True,
        global_step_volatility_page=True,
        global_distance_metrics=True,
        area_core_page=True,
        area_puzzle_page=True,
        area_vote_mode_alignment_page=True,
        area_group_opportunity_page=True,
        area_puzzle_gate_page=False,
        area_group_diagnostics_pages=True,
        area_learning_causal_page=True,
        area_assets_page=True,
        area_group_means_page=True,
        area_dist_to_ref_page=True,
    )


@dataclass(frozen=True)
class _ModelCfgParseResult:
    ok: bool
    model_cfg: dict[str, Any] | None
    source: Path | None
    error: str | None


def _parse_model_cfg_for_run(*, run_dir: Path) -> _ModelCfgParseResult:
    """Typed parse result for model section in config_used.yaml."""
    candidates = [
        run_dir / "config_used.yaml",
        run_dir.parent / "config_used.yaml",
    ]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        tried = ", ".join(str(p) for p in candidates)
        return _ModelCfgParseResult(
            ok=False,
            model_cfg=None,
            source=None,
            error=f"Missing config_used.yaml (tried: {tried})",
        )

    cfg_path = existing[0]
    try:
        raw_text = cfg_path.read_text(encoding="utf-8")
    except OSError as e:
        return _ModelCfgParseResult(
            ok=False,
            model_cfg=None,
            source=cfg_path,
            error=f"Failed to read config file {cfg_path}: {e}",
        )

    try:
        cfg = yaml.safe_load(raw_text)
    except yaml.YAMLError as e:
        return _ModelCfgParseResult(
            ok=False,
            model_cfg=None,
            source=cfg_path,
            error=f"Failed to parse YAML in {cfg_path}: {e}",
        )

    if not isinstance(cfg, dict):
        return _ModelCfgParseResult(
            ok=False,
            model_cfg=None,
            source=cfg_path,
            error=f"Invalid config root in {cfg_path}: expected mapping",
        )

    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        return _ModelCfgParseResult(
            ok=False,
            model_cfg=None,
            source=cfg_path,
            error=f"Invalid or missing 'model' section in {cfg_path}",
        )

    return _ModelCfgParseResult(
        ok=True,
        model_cfg=dict(model_cfg),
        source=cfg_path,
        error=None,
    )


def _load_model_cfg_for_run(*, run_dir: Path) -> dict[str, Any]:
    """Load model section from config_used.yaml (strict fail-fast)."""
    parsed = _parse_model_cfg_for_run(run_dir=run_dir)
    if not parsed.ok or parsed.model_cfg is None:
        raise RuntimeError(parsed.error or "Failed to parse model config")
    return parsed.model_cfg


def _load_required_finite_float_for_run(*, run_dir: Path, field: str) -> float:
    """Load required model float field from config_used.yaml (strict)."""
    model_cfg = _load_model_cfg_for_run(run_dir=run_dir)
    if field not in model_cfg:
        raise RuntimeError(f"Missing required model config field '{field}' in config_used.yaml")
    raw = model_cfg[field]
    try:
        val = float(raw)
    except (TypeError, ValueError) as e:
        raise RuntimeError(f"Invalid model config field '{field}': {raw!r}") from e
    if not np.isfinite(val):
        raise RuntimeError(f"Invalid non-finite model config field '{field}': {raw!r}")
    return val


def _load_required_bool_for_run(*, run_dir: Path, field: str) -> bool:
    """Load required model bool field from config_used.yaml (strict)."""
    model_cfg = _load_model_cfg_for_run(run_dir=run_dir)
    if field not in model_cfg:
        raise RuntimeError(f"Missing required model config field '{field}' in config_used.yaml")
    raw = model_cfg[field]
    if not isinstance(raw, bool):
        raise RuntimeError(f"Invalid model config field '{field}': expected bool, got {type(raw).__name__}")
    return raw


def _load_participation_alpha_for_run(*, run_dir: Path) -> float:
    """Load model.participation_alpha from config_used.yaml (strict)."""
    return _load_required_finite_float_for_run(run_dir=run_dir, field="participation_alpha")


def _load_altruism_alpha_for_run(*, run_dir: Path) -> float:
    """Load model.altruism_alpha from config_used.yaml (strict)."""
    return _load_required_finite_float_for_run(run_dir=run_dir, field="altruism_alpha")


def _load_participation_signal_mode_for_run(*, run_dir: Path) -> str:
    """Load model.participation_signal_mode from config_used.yaml (strict)."""
    model_cfg = _load_model_cfg_for_run(run_dir=run_dir)
    field = "participation_signal_mode"
    if field not in model_cfg:
        raise RuntimeError(f"Missing required model config field '{field}' in config_used.yaml")
    raw = model_cfg[field]
    if not isinstance(raw, str):
        raise RuntimeError(f"Invalid model config field '{field}': expected str, got {type(raw).__name__}")
    value = raw.strip()
    if not value:
        raise RuntimeError(f"Invalid model config field '{field}': empty string")
    return value


def _load_participation_signal_group_shrink_k_for_run(*, run_dir: Path) -> float:
    """Load model.participation_signal_group_shrink_k from config_used.yaml (strict)."""
    return _load_required_finite_float_for_run(run_dir=run_dir, field="participation_signal_group_shrink_k")


def _load_altruism_learning_for_run(*, run_dir: Path) -> bool:
    """Load model.altruism_learning from config_used.yaml (strict)."""
    return _load_required_bool_for_run(run_dir=run_dir, field="altruism_learning")


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
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            warnings.warn(
                f"Reference cache read failed at {cache_path}: {exc}. Recomputing reference payload.",
                RuntimeWarning,
                stacklevel=2,
            )

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


def _build_summary_stats(
    *,
    global_series: pd.DataFrame,
    area_series: pd.DataFrame,
    meta: dict[str, Any],
    static: dict[str, Any],
) -> dict[str, Any]:
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
            "turnout_mean": time_mean(global_series["turnout"].to_numpy(dtype=float)),
            "turnout_final": _safe_final(global_series["turnout"]),
            "turnout_volatility": step_volatility_l1_normalized(
                global_series["turnout"].to_numpy(dtype=float),
                value_range=100.0,
            ),
            "gini_assets_mean": time_mean(global_series["gini_assets"].to_numpy(dtype=float)),
            "gini_assets_final": _safe_final(global_series["gini_assets"]),
            "gini_assets_volatility": step_volatility_l1_normalized(
                global_series["gini_assets"].to_numpy(dtype=float),
                value_range=100.0,
            ),
            "gini_dissatisfaction_mean": time_mean(global_series["gini_dissatisfaction"].to_numpy(dtype=float)),
            "gini_dissatisfaction_final": _safe_final(global_series["gini_dissatisfaction"]),
            "gini_dissatisfaction_volatility": step_volatility_l1_normalized(
                global_series["gini_dissatisfaction"].to_numpy(dtype=float),
                value_range=100.0,
            ),
            "mean_dissatisfaction_mean": time_mean(global_series["mean_dissatisfaction"].to_numpy(dtype=float)),
            "mean_dissatisfaction_final": _safe_final(global_series["mean_dissatisfaction"]),
            "dist_to_reality_mean": time_mean(global_series["dist_to_reality"].to_numpy(dtype=float)),
            "dist_to_reality_final": _safe_final(global_series["dist_to_reality"]),
            "dist_to_reality_volatility": step_volatility_l1_normalized(
                global_series["dist_to_reality"].to_numpy(dtype=float),
                value_range=1.0,
            ),
            "diversity_entropy_mean": time_mean(global_series["diversity_first_choice_entropy"].to_numpy(dtype=float)),
            "diversity_entropy_final": _safe_final(global_series["diversity_first_choice_entropy"]),
        },
    }
    return summary


def _render_static_overview_pdf(*, out_pdf: Path, static: dict[str, Any], meta: dict[str, Any]) -> None:
    with PdfPages(out_pdf) as pdf:
        _append_static_overview_pages(pdf=pdf, static=static, meta=meta)


def _append_static_overview_pages(*, pdf: PdfPages, static: dict[str, Any], meta: dict[str, Any]) -> None:
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
        # Background: per-group preference-order stripes (same idea as ordering bands in area dist_to_reality plots).
        if personality_groups.ndim == 2 and personality_groups.shape[0] >= n_groups:
            n_slots = int(min(num_colors, personality_groups.shape[1]))
            for gi in range(n_groups):
                order = personality_groups[gi].astype(int).tolist()
                for rank, color_id in enumerate(order[:n_slots]):
                    y0 = 1.0 - float(rank + 1) / float(n_slots)
                    ax_global.add_patch(
                        plt.Rectangle(
                            (float(gi) - 0.4, y0),
                            0.8,
                            1.0 / float(n_slots),
                            facecolor=_sim_color(color_id),
                            edgecolor="none",
                            alpha=0.26,
                            zorder=0,
                        )
                    )
        # Foreground: transparent bars (black frames only).
        ax_global.bar(
            x,
            global_dist,
            width=0.8,
            facecolor="none",
            edgecolor=[get_group_color(i) for i in range(n_groups)],
            linewidth=1.2,
            zorder=2,
        )
        # Label shares at bar tops; if there is no space above, place just below.
        for gi, val in enumerate(global_dist.tolist()):
            y = float(val)
            label = f"{100.0 * y:.1f}%"
            if y <= 0.93:
                ax_global.text(
                    float(gi),
                    y + 0.02,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=get_group_color(int(gi)),
                    fontweight="bold",
                )
            else:
                ax_global.text(
                    float(gi),
                    y - 0.03,
                    label,
                    ha="center",
                    va="top",
                    fontsize=8,
                    color=get_group_color(int(gi)),
                    fontweight="bold",
                )
        ax_global.set_xticks(x)
        ax_global.set_xticklabels([f"g{i}" for i in range(n_groups)])
        # Encode group-color mapping directly in the x-axis labels.
        for gi, tick in enumerate(ax_global.get_xticklabels()):
            c = get_group_color(int(gi))
            r, g, b, _ = to_rgba(c)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            txt = "black" if luminance > 0.55 else "white"
            tick.set_color(txt)
            tick.set_bbox(
                dict(
                    facecolor=c,
                    edgecolor="none",  # frameless colored square-ish tag
                    boxstyle="round,pad=0.20,rounding_size=0.08",
                    alpha=0.95,
                )
            )
        _set_unit_ylim_visible(ax_global)
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
    ax_global.set_title("Personality Groups with their Global Shares")
    ax_global.set_yticks([])
    ax_global.set_ylabel("")
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


def _render_combined_global_summary_pdf(
    *,
    out_pdf: Path,
    run_dir: Path,
    global_series: pd.DataFrame,
    steps: pd.DataFrame,
    static: dict[str, Any],
    meta: dict[str, Any],
    refs_global: dict[str, np.ndarray | None],
    render_profile: _SummaryRenderProfile,
) -> None:
    with PdfPages(out_pdf) as pdf:
        if render_profile.global_colors_and_grids:
            _render_global_colors_and_grids_page(
                pdf=pdf,
                run_dir=run_dir,
                global_series=global_series,
                steps=steps,
                static=static,
                refs_global=refs_global,
            )
        if render_profile.global_static_overview:
            _append_static_overview_pages(pdf=pdf, static=static, meta=meta)
        if render_profile.global_per_area_group_distribution:
            _append_per_area_group_distribution_pages(pdf=pdf, static=static, meta=meta)
        if render_profile.global_core_metrics:
            _render_global_core_metrics_page(pdf=pdf, global_series=global_series, meta=meta, static=static)
        if render_profile.global_step_volatility_page:
            _render_global_step_volatility_page(pdf=pdf, global_series=global_series, meta=meta, static=static)
        if render_profile.global_distance_metrics:
            _render_global_distance_page(pdf=pdf, global_series=global_series)


def _append_per_area_group_distribution_pages(*, pdf: PdfPages, static: dict[str, Any], meta: dict[str, Any]) -> None:
    """Append area-wise group-distribution pages using the same visual style as global."""
    num_colors = int(static.get("num_colors", 0))
    info = static.get("personality_group_info", {}) or {}
    personality_groups = np.asarray(info.get("personality_groups", []), dtype=int)
    areas_info = info.get("areas", {}) or {}
    if personality_groups.ndim != 2:
        return
    n_groups = int(personality_groups.shape[0])
    if n_groups <= 0 or not isinstance(areas_info, dict) or len(areas_info) == 0:
        return

    rows: list[tuple[int, int, np.ndarray]] = []
    for area_key in sorted(areas_info.keys(), key=lambda x: int(x)):
        payload = areas_info.get(area_key) or {}
        dist = np.asarray(payload.get("personality_group_distribution", []), dtype=float)
        if dist.size != n_groups:
            continue
        rows.append((int(area_key), int(payload.get("num_agents", 0)), dist))
    if not rows:
        return

    run_seed = int(meta["run"]["run_seed"])
    rule_name = meta["run"].get("rule_name")
    per_page = 9
    n_pages = int(np.ceil(len(rows) / per_page))

    for p in range(n_pages):
        chunk = rows[p * per_page:(p + 1) * per_page]
        fig, axes = plt.subplots(3, 3, figsize=(11.69, 8.27))
        ax_list = axes.ravel()
        for idx, ax in enumerate(ax_list):
            if idx >= len(chunk):
                ax.axis("off")
                continue
            area_id, n_agents, dist = chunk[idx]
            x = np.arange(n_groups)
            # Background ordering stripes per group.
            n_slots = int(min(num_colors, personality_groups.shape[1]))
            for gi in range(n_groups):
                order = personality_groups[gi].astype(int).tolist()
                for rank, color_id in enumerate(order[:n_slots]):
                    y0 = 1.0 - float(rank + 1) / float(n_slots)
                    ax.add_patch(
                        plt.Rectangle(
                            (float(gi) - 0.4, y0),
                            0.8,
                            1.0 / float(n_slots),
                            facecolor=_sim_color(color_id),
                            edgecolor="none",
                            alpha=0.26,
                            zorder=0,
                        )
                    )
            # Transparent bars with black frame.
            ax.bar(
                x,
                dist,
                width=0.8,
                facecolor="none",
                edgecolor=[get_group_color(i) for i in range(n_groups)],
                linewidth=1.1,
                zorder=2,
            )
            # Percent labels.
            for gi, v in enumerate(dist.tolist()):
                y = float(v)
                txt = f"{100.0 * y:.0f}%"
                if y <= 0.92:
                    ax.text(
                        float(gi),
                        y + 0.02,
                        txt,
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color=get_group_color(int(gi)),
                        fontweight="bold",
                    )
                else:
                    ax.text(
                        float(gi),
                        y - 0.03,
                        txt,
                        ha="center",
                        va="top",
                        fontsize=7,
                        color=get_group_color(int(gi)),
                        fontweight="bold",
                    )
            ax.set_title(f"Area {area_id} (n={n_agents})", fontsize=9)
            _set_unit_ylim_visible(ax)
            ax.set_yticks([])
            ax.set_xticks(x)
            ax.set_xticklabels([f"g{i}" for i in range(n_groups)], fontsize=7)
            for gi, tick in enumerate(ax.get_xticklabels()):
                c = get_group_color(int(gi))
                r, g, b, _ = to_rgba(c)
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                txt = "black" if luminance > 0.55 else "white"
                tick.set_color(txt)
                tick.set_bbox(
                    dict(
                        facecolor=c,
                        edgecolor="none",
                        boxstyle="round,pad=0.14,rounding_size=0.06",
                        alpha=0.95,
                    )
                )
            ax.grid(True, axis="y", alpha=0.25)

        fig.suptitle(
            f"Per-Area Personality Group Distributions | run_seed={run_seed} | rule={rule_name} | page {p + 1}/{n_pages}",
            fontsize=11,
        )
        fig.tight_layout()
        pdf.savefig(fig, dpi=140)
        plt.close(fig)


def _render_area_detail_pdfs(
    *,
    run_dir: Path,
    out_dir: Path,
    area_series: pd.DataFrame,
    area_group_series: pd.DataFrame,
    static: dict[str, Any],
    meta: dict[str, Any],
    refs_by_area: dict[int, dict[str, np.ndarray | None]] | None = None,
    render_profile: _SummaryRenderProfile,
) -> None:
    area_ids = sorted(set(int(v) for v in area_series["area_id"].dropna().tolist()))
    for area_id in area_ids:
        block = area_series[area_series["area_id"].astype(int) == int(area_id)].sort_values("step").reset_index(drop=True)
        group_block = area_group_series[area_group_series["area_id"].astype(int) == int(area_id)].sort_values(["step", "group_idx"]).reset_index(drop=True)
        out_pdf = out_dir / f"area_{int(area_id)}.pdf"
        _render_area_detail_pdf(
            run_dir=run_dir,
            out_pdf=out_pdf,
            area_id=int(area_id),
            area_series=block,
            area_group_series=group_block,
            static=static,
            meta=meta,
            refs_area=(refs_by_area or {}).get(int(area_id), {}),
            render_profile=render_profile,
        )


def _render_area_detail_pdf(
    *,
    run_dir: Path,
    out_pdf: Path,
    area_id: int,
    area_series: pd.DataFrame,
    area_group_series: pd.DataFrame,
    static: dict[str, Any],
    meta: dict[str, Any],
    refs_area: dict[str, np.ndarray | None],
    render_profile: _SummaryRenderProfile,
) -> None:
    with PdfPages(out_pdf) as pdf:
        puzzle_threshold = _load_required_finite_float_for_run(
            run_dir=run_dir,
            field="break_even_distance_common",
        )

        x = area_series["step"].to_numpy(dtype=float)
        participants = area_series["participants"].to_numpy(dtype=float)
        eligible = area_series["eligible_voters"].to_numpy(dtype=float)
        turnout = area_series["turnout"].to_numpy(dtype=float)

        # Compact static info block for area context.
        areas_info = (((static.get("personality_group_info", {}) or {}).get("areas", {})) or {})
        area_info = areas_info.get(str(int(area_id)), {}) if isinstance(areas_info, dict) else {}
        area_n = int(area_info.get("num_agents", -1)) if isinstance(area_info, dict) else -1
        pg_dist = np.asarray(area_info.get("personality_group_distribution", []), dtype=float) if isinstance(area_info, dict) else np.asarray([], dtype=float)
        personality_groups = np.asarray(
            ((static.get("personality_group_info", {}) or {}).get("personality_groups", [])),
            dtype=int,
        )
        majority_txt = "n/a"
        if pg_dist.size > 0 and np.isfinite(pg_dist).any():
            gidx = int(np.nanargmax(pg_dist))
            majority_txt = f"g{gidx} ({100.0 * float(pg_dist[gidx]):.1f}%)"
        suptitle = (
            f"Area {area_id} Detail | run_seed={meta['run']['run_seed']} | "
            f"rule={meta['run'].get('rule_name')} | area_agents={area_n} | majority_group={majority_txt}"
        )
        current_rule_idx = int(((meta.get("run", {}) or {}).get("rule_idx", -1)) or -1)
        power_dirs = _compute_area_power_direction_orderings(
            area_group_series=area_group_series,
            personality_groups=personality_groups,
            num_colors=int(static.get("num_colors", 0)),
            meta=meta,
        )
        power_current = None
        for item in power_dirs:
            if int(item.get("rule_idx", -1)) == current_rule_idx:
                power_current = np.asarray(item.get("color_ordering", []), dtype=np.int64)
                break
        dist_decomp = _compute_area_puzzle_power_distances(
            area_series=area_series,
            num_colors=int(static.get("num_colors", 0)),
            power_ordering_current_rule=power_current,
            meta=meta,
        )

        # Page 1: left narrow reference/context + right dynamics panels.
        fig2 = plt.figure(figsize=(11.69, 8.27), constrained_layout=True)
        gs2 = fig2.add_gridspec(
            2,
            2,
            width_ratios=[1.0, 5.0],
            height_ratios=[1.0, 1.0],
            hspace=0.28,
            wspace=0.12,
        )
        ax_ref = fig2.add_subplot(gs2[0, 0])
        ax_color = fig2.add_subplot(gs2[0, 1])
        ax_pg = fig2.add_subplot(gs2[1, 0])
        ax_dist = fig2.add_subplot(gs2[1, 1], sharex=ax_color)
        color_cols = sorted(
            [c for c in area_series.columns if c.startswith("area_color_")],
            key=lambda n: int(n.split("_")[-1]),
        )
        refs_panel = {
            "utilitarian": refs_area.get("dist_to_ref_utilitarian"),
            "nash": refs_area.get("dist_to_ref_nash"),
            "egalitarian": refs_area.get("dist_to_ref_egalitarian"),
            "rawlsian": refs_area.get("dist_to_ref_rawlsian"),
        }
        _draw_reference_optima_panel(ax=ax_ref, refs=refs_panel, num_colors=int(static.get("num_colors", 0)))

        for i, c in enumerate(color_cols):
            ax_color.plot(x, area_series[c].to_numpy(dtype=float), color=_sim_color(i), label=f"color_{i}")
        ax_color.set_title("Area Color Distribution Curves")
        ax_color.set_ylabel("share")
        _set_unit_ylim_visible(ax_color)
        if color_cols:
            ax_color.legend(loc="best", fontsize=8, ncol=min(4, len(color_cols)))

        _draw_area_personality_group_distribution(
            ax=ax_pg,
            pg_dist=pg_dist,
            personality_groups=personality_groups,
            num_colors=int(static.get("num_colors", 0)),
        )

        # Background encodes elected ordering per step as stacked color bands
        # (top=rank 1 color ... bottom=last rank color).
        if "winning_option_id" in area_series.columns:
            ordering_bg = _build_elected_ordering_background_image(
                winning_option_ids=area_series["winning_option_id"].to_numpy(dtype=int),
                num_colors=int(static.get("num_colors", 0)),
            )
            if ordering_bg is not None:
                x0 = float(np.min(x)) - 0.5 if x.size > 0 else -0.5
                x1 = float(np.max(x)) + 0.5 if x.size > 0 else 0.5
                ax_dist.imshow(
                    ordering_bg,
                    origin="upper",
                    aspect="auto",
                    extent=[x0, x1, 0.0, 1.0],
                    interpolation="nearest",
                    zorder=0,
                )
        ax_dist.plot(
            x,
            area_series["dist_to_reality"].to_numpy(dtype=float),
            color="black",
            linestyle="--",
            linewidth=1.6,
            zorder=3,
        )
        ax_dist.set_title("dist_to_reality")
        ax_dist.set_ylabel("distance [0..1]")
        _set_unit_ylim_visible(ax_dist)
        for a in (ax_color, ax_dist):
            a.grid(True, alpha=0.25)
            a.set_xlabel("step")
        fig2.suptitle(suptitle, fontsize=11)
        if render_profile.area_core_page:
            pdf.savefig(fig2, dpi=140)
        plt.close(fig2)

        # Page 2: puzzle tracking (area-local puzzle distribution + puzzle distance).
        fig2p = plt.figure(figsize=(11.69, 8.27), constrained_layout=True)
        gs2p = fig2p.add_gridspec(
            2,
            2,
            width_ratios=[1.0, 5.0],
            height_ratios=[1.0, 1.0],
            hspace=0.28,
            wspace=0.12,
        )
        ax_power = fig2p.add_subplot(gs2p[0, 0])
        ax_pcurve = fig2p.add_subplot(gs2p[0, 1], sharex=ax_color)
        ax_pg_p = fig2p.add_subplot(gs2p[1, 0])
        ax_pdist = fig2p.add_subplot(gs2p[1, 1], sharex=ax_pcurve)
        _draw_power_direction_panel(
            ax=ax_power,
            power_dirs=power_dirs,
            current_rule_idx=current_rule_idx,
            num_colors=int(static.get("num_colors", 0)),
        )

        puzzle_cols = sorted(
            [c for c in area_series.columns if c.startswith("puzzle_color_")],
            key=lambda n: int(n.split("_")[-1]),
        )
        if puzzle_cols:
            for i, c in enumerate(puzzle_cols):
                ax_pcurve.plot(x, area_series[c].to_numpy(dtype=float), color=_sim_color(i), label=f"color_{i}")
            ax_pcurve.set_title("Puzzle Distribution Curves")
            ax_pcurve.set_ylabel("share")
            _set_unit_ylim_visible(ax_pcurve)
            ax_pcurve.legend(loc="best", fontsize=8, ncol=min(4, len(puzzle_cols)))
        else:
            ax_pcurve.axis("off")
            ax_pcurve.set_title("Puzzle Distribution Curves")
            ax_pcurve.text(
                0.5,
                0.5,
                "Puzzle distribution not logged for this run.",
                ha="center",
                va="center",
                fontsize=10,
            )

        _draw_area_personality_group_distribution(
            ax=ax_pg_p,
            pg_dist=pg_dist,
            personality_groups=personality_groups,
            num_colors=int(static.get("num_colors", 0)),
        )

        if "winning_option_id" in area_series.columns:
            ordering_bg = _build_elected_ordering_background_image(
                winning_option_ids=area_series["winning_option_id"].to_numpy(dtype=int),
                num_colors=int(static.get("num_colors", 0)),
            )
            if ordering_bg is not None:
                x0 = float(np.min(x)) - 0.5 if x.size > 0 else -0.5
                x1 = float(np.max(x)) + 0.5 if x.size > 0 else 0.5
                ax_pdist.imshow(
                    ordering_bg,
                    origin="upper",
                    aspect="auto",
                    extent=[x0, x1, 0.0, 1.0],
                    interpolation="nearest",
                    zorder=0,
                )
        ax_pdist.plot(
            x,
            area_series.get("puzzle_distance", pd.Series(np.nan, index=area_series.index)).to_numpy(dtype=float),
            color="black",
            linestyle="--",
            linewidth=1.6,
            label="outcome↔puzzle",
            zorder=3,
        )
        if np.isfinite(dist_decomp["dist_outcome_power"]).any():
            ax_pdist.plot(
                x,
                dist_decomp["dist_outcome_power"].astype(float),
                color="tab:red",
                linewidth=1.4,
                label="outcome↔power",
                zorder=3,
            )
        if np.isfinite(puzzle_threshold):
            ax_pdist.axhline(
                puzzle_threshold,
                color="#4a4a4a",
                linestyle=":",
                linewidth=1.4,
                label="threshold",
                zorder=2,
            )
        ax_pdist.set_title("Puzzle Distance vs Outcome / Power")
        ax_pdist.set_ylabel("distance [0..1]")
        _set_unit_ylim_visible(ax_pdist)
        if len(ax_pdist.lines) > 0:
            ax_pdist.legend(loc="upper right", fontsize=7, ncol=2)
        for a in (ax_pcurve, ax_pdist):
            if a.has_data():
                a.grid(True, alpha=0.25)
                a.set_xlabel("step")
        fig2p.suptitle(suptitle, fontsize=11)
        if render_profile.area_puzzle_page:
            pdf.savefig(fig2p, dpi=140)
        plt.close(fig2p)

        # Page 2b: use the newer puzzle/power decomposition page (formerly rendered later),
        # replacing the legacy split-decomposition page to avoid duplicate content.
        if render_profile.area_puzzle_page:
            _render_area_puzzle_gate_page(
                pdf=pdf,
                area_series=area_series,
                dist_decomp=dist_decomp,
                suptitle=suptitle,
                include_overview=False,
                include_decomposition=True,
            )

        # Page 3: vote-mode alignment diagnostics (with support/coverage context).
        fig2m, axes2m = plt.subplots(
            3,
            1,
            figsize=(11.69, 8.27),
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.0, 0.75]},
        )
        axm = np.asarray(axes2m).ravel()
        has_mode_alignment = False
        has_mode_coverage = False
        for plot_idx, series_spec, title in (
            (
                0,
                (
                    ("altruistic_rank1_match_puzzle_share", "altruistic_votes_count", "tab:green", "-", "altruistic -> puzzle"),
                    ("non_altruistic_rank1_match_puzzle_share", "non_altruistic_votes_count", "tab:red", "--", "non-altruistic -> puzzle"),
                ),
                "Rank-1 Match to Puzzle by Vote Mode [% of mode votes]",
            ),
            (
                1,
                (
                    ("altruistic_rank1_match_outcome_share", "altruistic_votes_count", "tab:green", "-", "altruistic -> elected"),
                    ("non_altruistic_rank1_match_outcome_share", "non_altruistic_votes_count", "tab:red", "--", "non-altruistic -> elected"),
                ),
                "Rank-1 Match to Elected Outcome by Vote Mode [% of mode votes]",
            ),
        ):
            for value_col, count_col, color, ls, label in series_spec:
                if value_col not in area_series.columns:
                    continue
                y = area_series[value_col].to_numpy(dtype=float)
                if not np.isfinite(y).any():
                    continue
                axm[plot_idx].plot(x, y, color=color, linestyle=ls, linewidth=0.9, alpha=0.25)
                axm[plot_idx].plot(
                    x,
                    _rolling_mean_nan(y),
                    color=color,
                    linestyle=ls,
                    linewidth=1.6,
                    alpha=0.98,
                    label=label,
                )
                if count_col in area_series.columns:
                    cvals = area_series[count_col].to_numpy(dtype=float)
                    low_support = np.isfinite(cvals) & (cvals < float(_MODE_ALIGNMENT_LOW_SUPPORT_VOTES))
                    if np.any(low_support):
                        axm[plot_idx].fill_between(
                            x,
                            0.0,
                            100.0,
                            where=low_support,
                            color=color,
                            alpha=0.06,
                            linewidth=0.0,
                        )
                has_mode_alignment = True
            axm[plot_idx].set_title(title)
            axm[plot_idx].set_ylabel("%")
            _set_percent_ylim_visible(axm[plot_idx])
            if len(axm[plot_idx].lines) > 0:
                axm[plot_idx].legend(loc="best", fontsize=8)

        axm_cov = axm[2]
        axm_cov.set_title("Vote-mode Coverage (solid=mode share, black=count)")
        axm_cov.set_ylabel("% of votes")
        for col, color, label in (
            ("altruistic_vote_share", "tab:green", "altruistic vote share"),
            ("non_altruistic_vote_share", "tab:red", "non-altruistic vote share"),
        ):
            if col not in area_series.columns:
                continue
            y = area_series[col].to_numpy(dtype=float)
            if not np.isfinite(y).any():
                continue
            axm_cov.plot(x, y, color=color, linewidth=0.9, alpha=0.25)
            axm_cov.plot(x, _rolling_mean_nan(y), color=color, linewidth=1.5, alpha=0.95, label=label)
            has_mode_coverage = True
        _set_percent_ylim_visible(axm_cov)

        cov_rhs = None
        if "vote_count_total" in area_series.columns:
            votes_total = area_series["vote_count_total"].to_numpy(dtype=float)
            if np.isfinite(votes_total).any():
                cov_rhs = axm_cov.twinx()
                cov_rhs.plot(x, votes_total, color="black", linewidth=1.1, alpha=0.85, label="mode vote count")
                cov_rhs.axhline(
                    float(_MODE_ALIGNMENT_LOW_SUPPORT_VOTES),
                    color="black",
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.8,
                )
                cov_rhs.set_ylabel("count", color="black")
                cov_rhs.tick_params(axis="y", colors="black")
                has_mode_coverage = True

        if len(axm_cov.lines) > 0 or (cov_rhs is not None and len(cov_rhs.lines) > 0):
            h1, l1 = axm_cov.get_legend_handles_labels()
            if cov_rhs is not None:
                h2, l2 = cov_rhs.get_legend_handles_labels()
                axm_cov.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
            else:
                axm_cov.legend(loc="best", fontsize=8)

        if not has_mode_alignment and not has_mode_coverage:
            for a in axm:
                a.text(0.5, 0.5, "Mode alignment unavailable (requires votes + puzzle logging)", ha="center", va="center")
                a.set_yticks([])
        for a in axm:
            a.grid(True, alpha=0.25)
            a.set_xlabel("step")
        fig2m.suptitle(suptitle, fontsize=11)
        fig2m.tight_layout()
        if render_profile.area_vote_mode_alignment_page:
            pdf.savefig(fig2m, dpi=140)
        plt.close(fig2m)

        # Page 4: group puzzle opportunity alignment + compact divergence diagnostics.
        fig2g, axes2g = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=True)
        axg = np.asarray(axes2g).ravel()
        opp_df = _compute_group_puzzle_opportunity_distances(
            area_series=area_series,
            personality_groups=personality_groups,
            num_colors=int(static.get("num_colors", 0)),
            meta=meta,
        )
        groups_sorted = sorted(int(v) for v in area_group_series["group_idx"].dropna().unique().tolist()) if not area_group_series.empty else []
        group_resident_count_static: dict[int, float] = {}
        if not area_group_series.empty and groups_sorted:
            for g in groups_sorted:
                vals = area_group_series.loc[area_group_series["group_idx"] == g, "residents"].to_numpy(dtype=float)
                group_resident_count_static[int(g)] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else float("nan")

        def _opp_group_is_small(g: int) -> bool:
            cnt = group_resident_count_static.get(int(g), float("nan"))
            return bool(np.isfinite(cnt) and cnt < float(_SMALL_GROUP_MIN_RESIDENTS))

        def _opp_group_alpha(g: int, *, normal: float = 0.95, small: float = 0.14) -> float:
            return float(small if _opp_group_is_small(int(g)) else normal)

        has_small_opp_groups = any(_opp_group_is_small(int(g)) for g in groups_sorted)
        major_groups_sorted = [int(g) for g in groups_sorted if not _opp_group_is_small(int(g))]
        spread_groups_sorted = major_groups_sorted if major_groups_sorted else list(groups_sorted)

        has_opp = False
        if not opp_df.empty:
            xs = opp_df["step"].to_numpy(dtype=float)
            shown_opp_groups = 0
            for g in groups_sorted:
                c = f"group_{g}_puzzle_opp_dist"
                if c in opp_df.columns:
                    y = opp_df[c].to_numpy(dtype=float)
                    if np.isfinite(y).any():
                        color = get_group_color(int(g))
                        is_small = _opp_group_is_small(int(g))
                        axg[0].plot(
                            xs,
                            y,
                            color=color,
                            linewidth=0.8 if is_small else 0.9,
                            alpha=_opp_group_alpha(int(g), normal=0.22, small=0.06),
                        )
                        axg[0].plot(
                            xs,
                            _rolling_mean_nan(y),
                            color=color,
                            linewidth=1.3 if is_small else 1.5,
                            alpha=_opp_group_alpha(int(g), normal=0.98, small=0.18),
                            label=f"g{g}" if not is_small else "_nolegend_",
                        )
                        if not is_small:
                            shown_opp_groups += 1
                        has_opp = True
        axg[0].set_title("Group Opportunity Alignment to Puzzle [distance(group ordering, puzzle ordering)]")
        axg[0].set_ylabel("distance [0..1]\n(lower=more aligned)")
        _set_unit_ylim_visible(axg[0])
        if not opp_df.empty and shown_opp_groups > 0:
            axg[0].legend(loc="best", fontsize=8, ncol=min(5, len(axg[0].lines)))
        if has_small_opp_groups:
            axg[0].text(
                0.01,
                0.03,
                f"Groups with residents < {_SMALL_GROUP_MIN_RESIDENTS} are plotted transparent.",
                transform=axg[0].transAxes,
                fontsize=7,
                ha="left",
                va="bottom",
                alpha=0.85,
            )

        has_behavior = False
        opp_rhs = None
        if not area_group_series.empty and groups_sorted:
            steps_g = np.asarray(sorted(int(v) for v in area_group_series["step"].dropna().unique().tolist()), dtype=float)
            part_vals_all = area_group_series["participants"].to_numpy(dtype=float)
            non_alt_vals_all = area_group_series["non_altruistic_voters"].to_numpy(dtype=float)
            non_alt_share_vals = np.full(len(area_group_series), np.nan, dtype=np.float32)
            np.divide(
                100.0 * non_alt_vals_all,
                part_vals_all,
                out=non_alt_share_vals,
                where=part_vals_all > 0.0,
            )
            p_non_alt_share = (
                area_group_series.assign(
                    non_alt_share=non_alt_share_vals
                )
                .pivot(index="step", columns="group_idx", values="non_alt_share")
                .reindex(index=steps_g.astype(int), columns=spread_groups_sorted)
                .astype(float)
            )
            p_turn = (
                area_group_series.pivot(index="step", columns="group_idx", values="turnout")
                .reindex(index=steps_g.astype(int), columns=spread_groups_sorted)
                .astype(float)
            )
            non_alt_mat = p_non_alt_share.to_numpy(dtype=float)
            turnout_mat = p_turn.to_numpy(dtype=float)
            with np.errstate(invalid="ignore"):
                non_alt_range = np.nanmax(non_alt_mat, axis=1) - np.nanmin(non_alt_mat, axis=1)
                turnout_range = np.nanmax(turnout_mat, axis=1) - np.nanmin(turnout_mat, axis=1)
            non_alt_all_nan = np.all(~np.isfinite(non_alt_mat), axis=1)
            turnout_all_nan = np.all(~np.isfinite(turnout_mat), axis=1)
            non_alt_range[non_alt_all_nan] = np.nan
            turnout_range[turnout_all_nan] = np.nan
            if np.isfinite(non_alt_range).any():
                axg[1].plot(steps_g, non_alt_range, color="tab:red", linewidth=0.9, alpha=0.25, linestyle=":")
                axg[1].plot(
                    steps_g,
                    _rolling_mean_nan(non_alt_range),
                    color="tab:red",
                    linewidth=1.6,
                    alpha=0.95,
                    linestyle=":",
                    label="non-alt share range across groups",
                )
                has_behavior = True
            if np.isfinite(turnout_range).any():
                axg[1].plot(steps_g, turnout_range, color="tab:blue", linewidth=0.9, alpha=0.25, linestyle="-")
                axg[1].plot(
                    steps_g,
                    _rolling_mean_nan(turnout_range),
                    color="tab:blue",
                    linewidth=1.6,
                    alpha=0.95,
                    linestyle="-",
                    label="turnout range across groups",
                )
                has_behavior = True
        if not opp_df.empty:
            opp_cols = [f"group_{int(g)}_puzzle_opp_dist" for g in spread_groups_sorted if f"group_{int(g)}_puzzle_opp_dist" in opp_df.columns]
            if opp_cols:
                opp_mat = opp_df[opp_cols].to_numpy(dtype=float)
                with np.errstate(invalid="ignore"):
                    opp_mean = np.nanmean(opp_mat, axis=1)
                    opp_range = np.nanmax(opp_mat, axis=1) - np.nanmin(opp_mat, axis=1)
                opp_all_nan = np.all(~np.isfinite(opp_mat), axis=1)
                opp_mean[opp_all_nan] = np.nan
                opp_range[opp_all_nan] = np.nan
                if np.isfinite(opp_mean).any() or np.isfinite(opp_range).any():
                    opp_rhs = axg[1].twinx()
                    if np.isfinite(opp_mean).any():
                        opp_rhs.plot(xs, _rolling_mean_nan(opp_mean), color="tab:green", linewidth=1.35, alpha=0.95, label="mean opp. dist")
                    if np.isfinite(opp_range).any():
                        opp_rhs.plot(xs, _rolling_mean_nan(opp_range), color="tab:purple", linewidth=1.25, alpha=0.95, linestyle="--", label="opp. dist range")
                    _set_unit_ylim_visible(opp_rhs)
                    opp_rhs.set_ylabel("puzzle-opportunity distance [0..1]", color="tab:green")
                    opp_rhs.tick_params(axis="y", colors="tab:green")
        axg[1].set_title("Cross-group Spread Diagnostics (behavior divergence + opportunity spread)")
        axg[1].set_ylabel("%")
        _set_percent_ylim_visible(axg[1])
        if len(axg[1].lines) > 0 or (opp_rhs is not None and len(opp_rhs.lines) > 0):
            h1, l1 = axg[1].get_legend_handles_labels()
            if opp_rhs is not None:
                h2, l2 = opp_rhs.get_legend_handles_labels()
                axg[1].legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=7, ncol=2)
            else:
                axg[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=7, ncol=2)
        if has_small_opp_groups:
            if major_groups_sorted:
                note = f"Spread excludes groups with residents < {_SMALL_GROUP_MIN_RESIDENTS}."
            else:
                note = f"All groups are < {_SMALL_GROUP_MIN_RESIDENTS}; spread uses all groups."
            axg[1].text(
                0.01,
                0.03,
                note,
                transform=axg[1].transAxes,
                fontsize=7,
                ha="left",
                va="bottom",
                alpha=0.85,
            )
        if not has_opp:
            axg[0].text(0.5, 0.5, "Opportunity alignment unavailable (requires puzzle distribution logging)", ha="center", va="center")
            axg[0].set_yticks([])
        if not has_behavior:
            axg[1].text(0.5, 0.5, "Group divergence series unavailable", ha="center", va="center")
            axg[1].set_yticks([])
        for a in axg:
            a.grid(True, alpha=0.25)
            a.set_xlabel("step")
        fig2g.suptitle(suptitle, fontsize=11)
        fig2g.tight_layout()
        if render_profile.area_group_opportunity_page:
            pdf.savefig(fig2g, dpi=140)
        plt.close(fig2g)

        if render_profile.area_puzzle_gate_page:
            _render_area_puzzle_gate_page(
                pdf=pdf,
                area_series=area_series,
                dist_decomp=dist_decomp,
                suptitle=suptitle,
                include_overview=True,
                include_decomposition=False,
            )

        # Following pages (group diagnostics etc.) come after the core area + puzzle analysis pages.
        if render_profile.area_group_diagnostics_pages and (not area_group_series.empty):
            _render_area_group_pages(
                pdf=pdf,
                area_group_series=area_group_series,
                x=x,
                turnout=turnout,
                participants=participants,
                eligible=eligible,
                suptitle=suptitle,
            )
        if render_profile.area_learning_causal_page and (not area_group_series.empty):
            _render_area_learning_causal_page(
                pdf=pdf,
                area_group_series=area_group_series,
                suptitle=suptitle,
            )

        # Page 10: gini assets + assets share by group
        fig1, axes1 = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=True)
        ax1 = np.asarray(axes1).ravel()
        ax1[0].plot(x, area_series["gini_assets"].to_numpy(dtype=float), color="tab:red")
        ax1[0].set_title("Gini Assets [0..100]")
        ax1[0].set_ylabel("gini")
        _set_percent_ylim_visible(ax1[0])
        assets_share_payload = _prepare_group_assets_share_series(area_group_series=area_group_series)
        if assets_share_payload is not None:
            steps_assets, groups_assets, p_assets_share = assets_share_payload
            for g in groups_assets:
                ax1[1].plot(
                    steps_assets,
                    p_assets_share[g].to_numpy(dtype=float),
                    color=get_group_color(int(g)),
                    linewidth=1.8,
                    label=f"g{g}",
                )
            ax1[1].set_title("Assets share by Group")
            ax1[1].set_ylabel("share")
            _set_unit_ylim_visible(ax1[1])
            if groups_assets:
                ax1[1].legend(loc="best", fontsize=8, ncol=min(5, len(groups_assets)))
        else:
            ax1[1].text(0.5, 0.5, "Assets share by group unavailable", ha="center", va="center")
            ax1[1].set_yticks([])

        for a in ax1:
            a.grid(True, alpha=0.25)
            a.set_xlabel("step")
        fig1.suptitle(suptitle, fontsize=11)
        fig1.tight_layout()
        if render_profile.area_assets_page:
            pdf.savefig(fig1, dpi=140)
        plt.close(fig1)

        # Then the rest: first group means page, then dist_to_ref + area means.
        if render_profile.area_group_means_page and (not area_group_series.empty):
            _render_area_group_means_page(
                pdf=pdf,
                area_group_series=area_group_series,
                x=x,
                gini_dissatisfaction=area_series["gini_dissatisfaction"].to_numpy(dtype=float),
            )

        fig3, axes3 = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=True)
        ax3 = np.asarray(axes3).ravel()
        for col, color, label in (
            ("dist_to_ref_utilitarian", "tab:blue", "utilitarian"),
            ("dist_to_ref_nash", "tab:purple", "nash"),
            ("dist_to_ref_egalitarian", "tab:orange", "egalitarian"),
            ("dist_to_ref_rawlsian", "tab:red", "rawlsian"),
        ):
            if col in area_series.columns:
                vals = area_series[col].to_numpy(dtype=float)
                if np.isfinite(vals).any():
                    ax3[0].plot(x, vals, color=color, label=label)
        ax3[0].set_title("dist_to_ref_*")
        ax3[0].set_ylabel("distance [0..1] (lower better)")
        _set_unit_ylim_visible(ax3[0])
        if len(ax3[0].lines) > 0:
            ax3[0].legend(loc="best", fontsize=8)

        area_means = _compute_area_weighted_means_from_group_series(area_group_series=area_group_series)
        if area_means is not None:
            ax3[1].plot(
                area_means["step"].to_numpy(dtype=float),
                area_means["mean_assets"].to_numpy(dtype=float),
                color="tab:blue",
                linewidth=1.6,
                label="mean_assets",
            )
            ax3b = ax3[1].twinx()
            ax3b.plot(
                area_means["step"].to_numpy(dtype=float),
                area_means["mean_dissatisfaction"].to_numpy(dtype=float),
                color="tab:orange",
                linestyle="--",
                linewidth=1.6,
                label="mean_dissatisfaction",
            )
            ax3[1].set_ylabel("assets", color="tab:blue")
            ax3b.set_ylabel("dissatisfaction [0..1]", color="tab:orange")
            _set_unit_ylim_visible(ax3b)
            ax3[1].tick_params(axis="y", colors="tab:blue")
            ax3b.tick_params(axis="y", colors="tab:orange")
            h1, l1 = ax3[1].get_legend_handles_labels()
            h2, l2 = ax3b.get_legend_handles_labels()
            if h1 or h2:
                ax3[1].legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
        else:
            ax3[1].text(0.5, 0.5, "Area mean assets/dissatisfaction unavailable", ha="center", va="center")
            ax3[1].set_yticks([])
        ax3[1].set_title("Area Mean Assets + Mean Dissatisfaction")
        for a in ax3:
            a.grid(True, alpha=0.25)
            a.set_xlabel("step")
        fig3.suptitle(suptitle, fontsize=11)
        fig3.tight_layout()
        if render_profile.area_dist_to_ref_page:
            pdf.savefig(fig3, dpi=140)
        plt.close(fig3)


def _render_area_group_pages(
    *,
    pdf: PdfPages,
    area_group_series: pd.DataFrame,
    x: np.ndarray,
    turnout: np.ndarray,
    participants: np.ndarray,
    eligible: np.ndarray,
    suptitle: str,
) -> None:
    groups = sorted(int(v) for v in area_group_series["group_idx"].dropna().unique().tolist())
    steps = np.asarray(sorted(int(v) for v in area_group_series["step"].dropna().unique().tolist()), dtype=float)
    if len(groups) == 0 or steps.size == 0:
        return

    pivot = lambda col: (
        area_group_series.pivot(index="step", columns="group_idx", values=col)
        .reindex(index=steps.astype(int), columns=groups)
        .astype(float)
    )
    pivot_optional = lambda col: pivot(col) if col in area_group_series.columns else None
    p_res = pivot("residents")
    p_elig = pivot("eligible")
    p_part = pivot("participants")
    p_non_alt = pivot("non_altruistic_voters")
    p_turn = pivot("turnout")
    p_assets = pivot("mean_assets")
    p_dissat = pivot("mean_dissatisfaction")
    p_res_share = pivot("resident_share")
    p_part_share = pivot("participant_share")
    p_part_delta = pivot_optional("participants_mean_delta_rel")
    p_abs_delta = pivot_optional("abstainers_mean_delta_rel")
    p_part_fee = pivot_optional("participants_mean_fee")
    p_part_fee_assets = pivot_optional("participants_mean_fee_over_assets")
    p_q_std_within = pivot_optional("group_std_q_participation")
    p_p_std_within = pivot_optional("group_std_participation_probability")
    p_gini_assets_within = pivot_optional("group_gini_assets_within")
    p_gini_diss_within = pivot_optional("group_gini_dissatisfaction_within")
    p_alt_a_update = pivot_optional("altruistic_voters_mean_altruism_update_proxy")
    p_non_alt_a_update = pivot_optional("non_altruistic_voters_mean_altruism_update_proxy")
    p_alt_a_delta_exact = pivot_optional("altruistic_voters_mean_altruism_delta")
    p_non_alt_a_delta_exact = pivot_optional("non_altruistic_voters_mean_altruism_delta")
    p_mode_switch = pivot_optional("vote_mode_switch_share")
    p_mode_switch_from_alt = pivot_optional("vote_mode_switch_from_altruistic_share")
    p_mode_switch_from_non_alt = pivot_optional("vote_mode_switch_from_non_altruistic_share")
    p_part_to_abs_switch = pivot_optional("participation_switch_to_abstain_share")
    p_alt_dsig = pivot_optional("altruistic_voters_mean_dissatisfaction_signal")
    p_non_alt_dsig = pivot_optional("non_altruistic_voters_mean_dissatisfaction_signal")

    # Page 2: top participants/eligible/non-altruistic; bottom composition share, both with right-side references.
    fig4 = plt.figure(figsize=(11.69, 8.27))
    gs4 = fig4.add_gridspec(2, 1, height_ratios=[1.0, 1.0])
    top = gs4[0].subgridspec(1, 2, width_ratios=[4.0, 0.15], wspace=0.01)
    bottom = gs4[1].subgridspec(1, 2, width_ratios=[4.0, 0.15], wspace=0.01)
    ax4_left = fig4.add_subplot(top[0, 0])
    ax4_ref = fig4.add_subplot(top[0, 1])
    ax4_bottom = fig4.add_subplot(bottom[0, 0])
    ax4_bottom_ref = fig4.add_subplot(bottom[0, 1])
    resident_share_static_vals: list[float] = []
    for g in groups:
        vals = area_group_series[area_group_series["group_idx"] == g]["resident_share"].dropna()
        resident_share_static_vals.append(float(vals.iloc[0]) if not vals.empty else 0.0)

    resident_share_static = np.asarray(resident_share_static_vals, dtype=float)
    group_resident_count_static: dict[int, float] = {}
    if groups:
        for g in groups:
            vals = p_res[g].to_numpy(dtype=float)
            group_resident_count_static[int(g)] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else float("nan")

    def _group_is_small(g: int) -> bool:
        cnt = group_resident_count_static.get(int(g), float("nan"))
        return bool(np.isfinite(cnt) and cnt < float(_SMALL_GROUP_MIN_RESIDENTS))

    def _group_alpha(g: int, *, normal: float = 0.9, small: float = 0.14) -> float:
        return float(small if _group_is_small(int(g)) else normal)

    has_small_groups = any(_group_is_small(int(g)) for g in groups)

    ax4_ref.set_xlim(float(np.min(steps)) if steps.size > 0 else 0.0, float(np.max(steps)) if steps.size > 0 else 1.0)
    max_res = 0.0
    if groups:
        max_res = float(np.nanmax([np.nanmax(p_res[g].to_numpy(dtype=float)) for g in groups]))
    ax4_ref.set_ylim(0.0, max(1.0, max_res * 1.05))
    ax4_ref.set_title("Total\nCount")
    ax4_ref.axis("off")

    for g in groups:
        color = get_group_color(int(g))
        ax4_left.plot(steps, p_part[g].to_numpy(dtype=float), color=color, linewidth=1.15, label=f"g{g} participants")
        ax4_left.plot(steps, p_elig[g].to_numpy(dtype=float), color=color, linestyle=":", alpha=0.85, linewidth=1.2)
        ax4_left.plot(
            steps,
            p_non_alt[g].to_numpy(dtype=float),
            color=color,
            linestyle="--",
            alpha=0.9,
            linewidth=1.2,
        )
    ax4_left.set_title("Participants (solid) + Eligible (dotted) + Non-altruistic voters (dashed) by Group")
    ax4_left.set_ylabel("count")
    ax4_left.grid(True, alpha=0.25)
    ax4_left.set_xlabel("step")
    if groups:
        ax4_left.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=8, ncol=min(5, len(groups)))
    ax4_left.set_ylim(bottom=0.0)
    ax4_left.margins(x=0.0, y=0.0)
    ax4_left.spines["bottom"].set_position(("data", 0.0))

    # Top reference: thin dashed residents-by-group traces (no axis).
    for g in groups:
        ax4_ref.plot(
            steps,
            p_res[g].to_numpy(dtype=float),
            color=get_group_color(int(g)),
            linestyle="--",
            linewidth=1.1,
            alpha=0.9,
        )
    left_stack = [p_part_share[g].to_numpy(dtype=float) for g in groups]
    if left_stack:
        ax4_bottom.stackplot(
            steps,
            *left_stack,
            labels=[f"g{g}" for g in groups],
            colors=[get_group_color(int(g)) for g in groups],
            alpha=0.9,
        )
    ax4_bottom.set_title("Participant Composition Share by Group")
    ax4_bottom.set_ylabel("share")
    ax4_bottom.set_ylim(0.0, 1.0)
    ax4_bottom.grid(True, alpha=0.25)
    ax4_bottom.set_xlabel("step")
    ax4_bottom.set_ylim(bottom=0.0)
    ax4_bottom.margins(x=0.0, y=0.0)
    ax4_bottom.spines["bottom"].set_position(("data", 0.0))

    ax4_bottom_ref.set_ylim(0.0, 1.0)
    ax4_bottom_ref.set_title("Total\nShare")
    ax4_bottom_ref.axis("off")
    bottom_share = 0.0
    for g, s in zip(groups, resident_share_static):
        ax4_bottom_ref.bar(0, s, bottom=bottom_share, width=0.2, color=get_group_color(int(g)), edgecolor="none")
        y_mid = bottom_share + (float(s) / 2.0)
        label = f"g{int(g)}"
        if float(s) >= 0.10:
            label = f"g{int(g)}\n{int(round(float(s) * 100.0))}%"
        r, gg, b, _ = to_rgba(get_group_color(int(g)))
        luminance = 0.299 * r + 0.587 * gg + 0.114 * b
        txt_color = "black" if luminance > 0.55 else "white"
        if float(s) >= 0.045:
            ax4_bottom_ref.text(
                0.0,
                y_mid,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color=txt_color,
                fontweight="bold",
            )
        bottom_share += s

    fig4.tight_layout()
    pdf.savefig(fig4, dpi=140)
    plt.close(fig4)

    # Page 3: non-altruistic share among participants by group (top), Turnout by Group (bottom).
    fig3, ax3 = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=True)
    ax3[0].set_title("Non-altruistic Share Among Participants by Group")
    ax3[0].set_ylabel("%")
    for g in groups:
        color = get_group_color(int(g))
        part_vals = p_part[g].to_numpy(dtype=float)
        non_alt_vals = p_non_alt[g].to_numpy(dtype=float)
        share = np.zeros_like(part_vals, dtype=float)
        np.divide(non_alt_vals, part_vals, out=share, where=part_vals > 0.0)
        ax3[0].plot(
            steps,
            share * 100.0,
            color=color,
            linestyle="-",
            linewidth=0.9,
            alpha=0.9,
            label=f"g{g}",
        )
    total_participants = p_part.sum(axis=1).to_numpy(dtype=float)
    total_non_alt = p_non_alt.sum(axis=1).to_numpy(dtype=float)
    total_share = np.zeros_like(total_non_alt, dtype=float)
    np.divide(total_non_alt, total_participants, out=total_share, where=total_participants > 0.0)
    ax3[0].plot(
        steps,
        total_share * 100.0,
        color="black",
        linestyle="--",
        linewidth=1.4,
        alpha=0.95,
        label="total non-altruistic share",
    )
    _set_percent_ylim_visible(ax3[0])
    if groups:
        ax3[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=7, ncol=min(5, len(groups) + 1))
    for g in groups:
        ax3[1].plot(steps, p_turn[g].to_numpy(dtype=float), color=get_group_color(int(g)), linewidth=0.9, label=f"g{g}")
    ax3[1].plot(x, turnout, color="black", linestyle="--", linewidth=1.4, alpha=0.9, label="turnout total")
    ax3[1].set_title("Turnout by Group [% of Residents]")
    ax3[1].set_ylabel("%")
    _set_percent_ylim_visible(ax3[1])
    for a in ax3:
        a.grid(True, alpha=0.25)
        a.set_xlabel("step")
    fig3.suptitle(suptitle + " | Group Diagnostics", fontsize=11)
    fig3.tight_layout()
    pdf.savefig(fig3, dpi=140)
    plt.close(fig3)

    # Page 4: top plot = non-altruistic group shares over total non-altruistic participants.
    fig5, ax5 = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=True)
    ax5[0].set_title("Non-altruistic Share by Group [% of Total Non-altruistic Participants]")
    ax5[0].set_ylabel("%")
    total_non_alt = p_non_alt.sum(axis=1).to_numpy(dtype=float)
    for g in groups:
        non_alt_vals = p_non_alt[g].to_numpy(dtype=float)
        share = np.zeros_like(non_alt_vals, dtype=float)
        np.divide(non_alt_vals, total_non_alt, out=share, where=total_non_alt > 0.0)
        ax5[0].plot(
            steps,
            share * 100.0,
            color=get_group_color(int(g)),
            linewidth=1.6,
            alpha=0.9,
            label=f"g{g}",
        )
    _set_percent_ylim_visible(ax5[0])
    if groups:
        ax5[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=7, ncol=min(5, len(groups)))
    ax5[0].grid(True, alpha=0.25)
    ax5[0].set_xlabel("step")

    ax5[1].set_title("Altruistic / Non-altruistic Share by Group [% of Total Participants]")
    ax5[1].set_ylabel("%")
    total_participants = p_part.sum(axis=1).to_numpy(dtype=float)
    for g in groups:
        part_vals = p_part[g].to_numpy(dtype=float)
        non_alt_vals = p_non_alt[g].to_numpy(dtype=float)
        alt_vals = np.maximum(0.0, part_vals - non_alt_vals)
        share = np.zeros_like(non_alt_vals, dtype=float)
        np.divide(non_alt_vals, total_participants, out=share, where=total_participants > 0.0)
        alt_share = np.zeros_like(alt_vals, dtype=float)
        np.divide(alt_vals, total_participants, out=alt_share, where=total_participants > 0.0)
        ax5[1].plot(
            steps,
            share * 100.0,
            color=get_group_color(int(g)),
            linewidth=1.6,
            alpha=0.9,
            label=f"g{g} non-alt",
        )
        ax5[1].plot(
            steps,
            alt_share * 100.0,
            color=get_group_color(int(g)),
            linestyle=":",
            linewidth=1.25,
            alpha=0.9,
            label=f"g{g} alt",
        )
    total_non_alt = p_non_alt.sum(axis=1).to_numpy(dtype=float)
    total_non_alt_share = np.zeros_like(total_non_alt, dtype=float)
    np.divide(total_non_alt, total_participants, out=total_non_alt_share, where=total_participants > 0.0)
    ax5[1].plot(
        steps,
        total_non_alt_share * 100.0,
        color="black",
        linestyle="--",
        linewidth=1.4,
        alpha=0.95,
        label="total non-alt share",
    )
    _set_percent_ylim_visible(ax5[1])
    if groups:
        ax5[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=7, ncol=min(6, len(groups) + 1))
    ax5[1].grid(True, alpha=0.25)
    ax5[1].set_xlabel("step")
    fig5.suptitle(suptitle + " | Group Diagnostics", fontsize=11)
    fig5.tight_layout()
    pdf.savefig(fig5, dpi=140)
    plt.close(fig5)

    # Page 5 (Page A): incentives and costs by group.
    fig6, ax6 = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=True)
    ax6[0].set_title("Mean Net Election Delta by Group (solid=participants, dotted=abstainers)")
    ax6[0].set_ylabel("delta_rel [%]")
    if p_part_delta is not None and p_abs_delta is not None:
        for g in groups:
            color = get_group_color(int(g))
            ax6[0].plot(
                steps,
                p_part_delta[g].to_numpy(dtype=float) * 100.0,
                color=color,
                linewidth=0.9,
                alpha=0.9,
                label=f"g{g}",
            )
            ax6[0].plot(
                steps,
                p_abs_delta[g].to_numpy(dtype=float) * 100.0,
                color=color,
                linestyle=":",
                linewidth=1.8,
                alpha=0.9,
            )
        if groups:
            ax6[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=7, ncol=min(5, len(groups)))
    else:
        ax6[0].text(0.5, 0.5, "Participant/abstainer delta data unavailable", ha="center", va="center")
        ax6[0].set_yticks([])

    ax6[1].set_title("Election Fee Share of Pool by Group (solid) + Fee/Assets (dotted)")
    ax6[1].set_ylabel("% of fee pool")
    if p_part_fee is not None and p_part_fee_assets is not None:
        ax6b = ax6[1].twinx()
        group_fee_total = p_part_fee * p_part
        fee_pool_total = group_fee_total.sum(axis=1).to_numpy(dtype=float)
        for g in groups:
            color = get_group_color(int(g))
            group_fee = group_fee_total[g].to_numpy(dtype=float)
            fee_share = np.zeros_like(group_fee, dtype=float)
            np.divide(group_fee, fee_pool_total, out=fee_share, where=fee_pool_total > 0.0)
            ax6[1].plot(
                steps,
                fee_share * 100.0,
                color=color,
                linewidth=0.9,
                alpha=0.9,
                label=f"g{g}",
            )
            ax6b.plot(
                steps,
                p_part_fee_assets[g].to_numpy(dtype=float) * 100.0,
                color=color,
                linestyle=":",
                linewidth=1.8,
                alpha=0.9,
            )
        _set_percent_ylim_visible(ax6[1])
        ax6b.set_ylabel("fee/assets [%]")
        if groups:
            ax6[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=7, ncol=min(5, len(groups)))
    else:
        ax6[1].text(0.5, 0.5, "Fee diagnostics unavailable", ha="center", va="center")
        ax6[1].set_yticks([])
    for a in ax6:
        a.grid(True, alpha=0.25)
        a.set_xlabel("step")
    fig6.suptitle(suptitle + " | Group Diagnostics", fontsize=11)
    fig6.tight_layout()
    pdf.savefig(fig6, dpi=140)
    plt.close(fig6)

    # Page 6 (Page B): participation dropout by group.
    fig7, ax7 = plt.subplots(1, 1, figsize=(11.69, 8.27), sharex=True)
    ax7.set_title("Participation Dropout Share by Group [% switched participating->abstaining]")
    ax7.set_ylabel("%")
    has_part_drop = (
        p_part_to_abs_switch is not None
        and np.isfinite(p_part_to_abs_switch.to_numpy(dtype=float)).any()
    )
    if has_part_drop:
        shown_groups = 0
        for g in groups:
            is_small = _group_is_small(int(g))
            ax7.plot(
                steps,
                p_part_to_abs_switch[g].to_numpy(dtype=float) * 100.0,
                color=get_group_color(int(g)),
                linewidth=0.8 if is_small else 1.0,
                alpha=_group_alpha(int(g), normal=0.9, small=0.16),
                label=f"g{g}" if not is_small else "_nolegend_",
            )
            if not is_small:
                shown_groups += 1
        _set_percent_ylim_visible(ax7)
        if shown_groups > 0:
            ax7.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=7, ncol=min(5, len(groups)))
        if has_small_groups:
            ax7.text(
                0.01,
                0.03,
                f"Groups with residents < {_SMALL_GROUP_MIN_RESIDENTS} are plotted transparent.",
                transform=ax7.transAxes,
                fontsize=7,
                ha="left",
                va="bottom",
                alpha=0.85,
            )
    else:
        ax7.text(0.5, 0.5, "Participation dropout share unavailable", ha="center", va="center")
        ax7.set_yticks([])
    ax7.grid(True, alpha=0.25)
    ax7.set_xlabel("step")
    fig7.suptitle(suptitle + " | Group Diagnostics", fontsize=11)
    fig7.tight_layout()
    pdf.savefig(fig7, dpi=140)
    plt.close(fig7)

    # Page 7b: within-group dispersion / inequality (q, p, assets, dissatisfaction).
    fig7b, ax7b = plt.subplots(2, 2, figsize=(11.69, 8.27), sharex=True)
    ax7b = np.asarray(ax7b).ravel()

    def _plot_group_panel(ax, pdata, *, title: str, ylab: str, scale: float = 1.0, percent: bool = False) -> None:
        has = pdata is not None and np.isfinite(pdata.to_numpy(dtype=float)).any()
        ax.set_title(title)
        ax.set_ylabel(ylab)
        if has and pdata is not None:
            for g in groups:
                ax.plot(
                    steps,
                    pdata[g].to_numpy(dtype=float) * scale,
                    color=get_group_color(int(g)),
                    linewidth=1.0,
                    alpha=0.9,
                    label=f"g{g}",
                )
            if percent:
                _set_percent_ylim_visible(ax)
            if groups:
                ax.legend(loc="best", fontsize=7, ncol=min(4, len(groups)))
        else:
            ax.text(0.5, 0.5, "Unavailable", ha="center", va="center")
            ax.set_yticks([])
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("step")

    _plot_group_panel(
        ax7b[0],
        p_q_std_within,
        title="Within-Group Std of q_participation",
        ylab="std(q)",
    )
    _plot_group_panel(
        ax7b[1],
        p_p_std_within,
        title="Within-Group Std of participation probability",
        ylab="std(p)",
    )
    _plot_group_panel(
        ax7b[2],
        p_gini_assets_within,
        title="Within-Group Gini Assets",
        ylab="gini [0..100]",
        percent=True,
    )
    _plot_group_panel(
        ax7b[3],
        p_gini_diss_within,
        title="Within-Group Gini Dissatisfaction",
        ylab="gini [0..100]",
        percent=True,
    )
    fig7b.suptitle(suptitle + " | Group Diagnostics", fontsize=11)
    fig7b.tight_layout()
    pdf.savefig(fig7b, dpi=140)
    plt.close(fig7b)

    # Page 9: vote-mode switching share views by group.
    fig9, ax9 = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=True)
    ax9[0].set_title("Share of Participants who switched Vote-Mode from their previous participation")
    ax9[0].set_ylabel("%")
    has_switch = (
        p_mode_switch is not None
        and p_mode_switch_from_alt is not None
        and p_mode_switch_from_non_alt is not None
        and (
            np.isfinite(p_mode_switch.to_numpy(dtype=float)).any()
            or np.isfinite(p_mode_switch_from_alt.to_numpy(dtype=float)).any()
            or np.isfinite(p_mode_switch_from_non_alt.to_numpy(dtype=float)).any()
        )
    )
    if has_switch:
        shown_groups_switch = 0
        for g in groups:
            is_small = _group_is_small(int(g))
            color = get_group_color(int(g))
            ax9[0].plot(
                steps,
                p_mode_switch[g].to_numpy(dtype=float) * 100.0,
                color=color,
                linewidth=0.9 if is_small else 1.2,
                linestyle="-",
                alpha=_group_alpha(int(g), normal=0.9, small=0.16),
                label=f"g{g}" if not is_small else "_nolegend_",
            )
            if not is_small:
                shown_groups_switch += 1
        _set_percent_ylim_visible(ax9[0])
        if shown_groups_switch > 0:
            ax9[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=7, ncol=min(5, len(groups)))
        if has_small_groups:
            ax9[0].text(
                0.01,
                0.03,
                f"Groups with residents < {_SMALL_GROUP_MIN_RESIDENTS} are plotted transparent.",
                transform=ax9[0].transAxes,
                fontsize=7,
                ha="left",
                va="bottom",
                alpha=0.85,
            )
    else:
        ax9[0].text(
            0.5,
            0.5,
            "Vote-mode switch share unavailable",
            ha="center",
            va="center",
        )
        ax9[0].set_yticks([])

    ax9[1].set_title("Vote-Mode Switch Origins by Group [% of participants: from altruistic / from non-altruistic]")
    ax9[1].set_ylabel("%")
    if has_switch:
        shown_groups_origins = 0
        for g in groups:
            is_small = _group_is_small(int(g))
            color = get_group_color(int(g))
            ax9[1].plot(
                steps,
                p_mode_switch_from_alt[g].to_numpy(dtype=float) * 100.0,
                color=color,
                linewidth=0.75 if is_small else 0.9,
                linestyle="-",
                alpha=_group_alpha(int(g), normal=0.9, small=0.16),
                label=f"g{g}" if not is_small else "_nolegend_",
            )
            ax9[1].plot(
                steps,
                p_mode_switch_from_non_alt[g].to_numpy(dtype=float) * 100.0,
                color=color,
                linewidth=1.0 if is_small else 1.5,
                linestyle=":",
                alpha=_group_alpha(int(g), normal=0.9, small=0.16),
                label="_nolegend_",
            )
            if not is_small:
                shown_groups_origins += 1
        _set_percent_ylim_visible(ax9[1])
        if shown_groups_origins > 0:
            group_leg = ax9[1].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.02),
                fontsize=7,
                ncol=min(5, shown_groups_origins),
                title="groups",
            )
            ax9[1].add_artist(group_leg)
            style_handles = [
                Line2D([0], [0], color="black", linestyle="-", linewidth=0.9, label="from altruistic"),
                Line2D([0], [0], color="black", linestyle=":", linewidth=1.35, label="from non-altruistic"),
            ]
            ax9[1].legend(
                handles=style_handles,
                loc="upper left",
                fontsize=7,
                title="line style",
            )
        if has_small_groups:
            ax9[1].text(
                0.01,
                0.03,
                f"Groups with residents < {_SMALL_GROUP_MIN_RESIDENTS} are plotted transparent.",
                transform=ax9[1].transAxes,
                fontsize=7,
                ha="left",
                va="bottom",
                alpha=0.85,
            )
    else:
        ax9[1].text(0.5, 0.5, "Vote-mode switch share unavailable", ha="center", va="center")
        ax9[1].set_yticks([])

    for a in ax9:
        a.grid(True, alpha=0.25)
        a.set_xlabel("step")
    fig9.suptitle(suptitle + " | Group Diagnostics", fontsize=11)
    fig9.tight_layout()
    pdf.savefig(fig9, dpi=140)
    plt.close(fig9)

    # Page 10: altruism signal + altruism shift proxy.
    fig10, ax10 = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=True)
    ax10[0].set_title("Altruism Learning Signal by Group (solid=altruistic voters, dotted=non-altruistic voters)")
    ax10[0].set_ylabel("signal (dissatisfaction)")
    has_alt_sig = (
        p_alt_dsig is not None
        and p_non_alt_dsig is not None
        and (
            np.isfinite(p_alt_dsig.to_numpy(dtype=float)).any()
            or np.isfinite(p_non_alt_dsig.to_numpy(dtype=float)).any()
        )
    )
    if has_alt_sig:
        for g in groups:
            color = get_group_color(int(g))
            ax10[0].plot(
                steps,
                p_alt_dsig[g].to_numpy(dtype=float),
                color=color,
                linewidth=0.9,
                alpha=0.9,
                label=f"g{g}",
            )
            ax10[0].plot(
                steps,
                p_non_alt_dsig[g].to_numpy(dtype=float),
                color=color,
                linestyle=":",
                linewidth=1.8,
                alpha=0.9,
            )
        ax10[0].axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.8)
        if groups:
            ax10[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=7, ncol=min(5, len(groups)))
    else:
        ax10[0].text(0.5, 0.5, "Altruism learning signal unavailable", ha="center", va="center")
        ax10[0].set_yticks([])

    alt_shift_plot = (p_alt_a_delta_exact, p_non_alt_a_delta_exact) if (
        p_alt_a_delta_exact is not None and p_non_alt_a_delta_exact is not None
    ) else (p_alt_a_update, p_non_alt_a_update)
    alt_shift_is_exact = p_alt_a_delta_exact is not None and p_non_alt_a_delta_exact is not None
    ax10[1].set_title(
        "Altruism Learning Shift by Group "
        f"({'exact Δa' if alt_shift_is_exact else 'proxy'}; solid=altruistic voters, dashed=non-altruistic voters)"
    )
    ax10[1].set_ylabel("Δaltruism" if alt_shift_is_exact else "delta_altruism proxy")
    has_alt_shift = (
        alt_shift_plot[0] is not None
        and alt_shift_plot[1] is not None
        and (
            np.isfinite(alt_shift_plot[0].to_numpy(dtype=float)).any()
            or np.isfinite(alt_shift_plot[1].to_numpy(dtype=float)).any()
        )
    )
    if has_alt_shift:
        for g in groups:
            color = get_group_color(int(g))
            ax10[1].plot(
                steps,
                alt_shift_plot[0][g].to_numpy(dtype=float),
                color=color,
                linewidth=1.8,
                alpha=0.9,
                label=f"g{g}",
            )
            ax10[1].plot(
                steps,
                alt_shift_plot[1][g].to_numpy(dtype=float),
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
            )
        ax10[1].axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.8)
        if groups:
            ax10[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=7, ncol=min(5, len(groups)))
        if not alt_shift_is_exact:
            ax10[1].text(
                0.01,
                0.03,
                "Proxy = altruism_alpha * dissatisfaction_signal (surprise_learning only). "
                "Exact Δa is available from altruism_factor snapshots.",
                transform=ax10[1].transAxes,
                fontsize=7,
                ha="left",
                va="bottom",
                alpha=0.85,
            )
    else:
        ax10[1].text(
            0.5,
            0.5,
            "Altruism learning shift proxy unavailable\n"
            "(likely altruism_learning=false or no mode-split voter coverage).",
            ha="center",
            va="center",
        )
        ax10[1].set_yticks([])
    for a in ax10:
        a.grid(True, alpha=0.25)
        a.set_xlabel("step")
    fig10.suptitle(suptitle + " | Group Diagnostics", fontsize=11)
    fig10.tight_layout()
    pdf.savefig(fig10, dpi=140)
    plt.close(fig10)

    # Group means page is rendered later from _render_area_detail_pdf after dist_to_ref.


def _render_area_learning_causal_page(
    *,
    pdf: PdfPages,
    area_group_series: pd.DataFrame,
    suptitle: str,
) -> None:
    groups = sorted(int(v) for v in area_group_series["group_idx"].dropna().unique().tolist())
    steps = np.asarray(sorted(int(v) for v in area_group_series["step"].dropna().unique().tolist()), dtype=float)
    if len(groups) == 0 or steps.size == 0:
        return

    pivot = lambda col: (
        area_group_series.pivot(index="step", columns="group_idx", values=col)
        .reindex(index=steps.astype(int), columns=groups)
        .astype(float)
    )
    pivot_optional = lambda col: pivot(col) if col in area_group_series.columns else None

    p_res = pivot("residents")
    p_mu = pivot_optional("group_mu_delta_rel")
    p_mu_bar = pivot_optional("global_mu_delta_rel")
    p_w = pivot_optional("group_signal_shrink_weight")
    p_r = pivot_optional("group_signal_component")
    p_part_sig = pivot_optional("participants_mean_participation_signal")
    p_abs_sig = pivot_optional("abstainers_mean_participation_signal")
    p_part_sig_group = pivot_optional("participants_mean_signal_group_component")
    p_part_sig_fee = pivot_optional("participants_mean_signal_fee_component")
    p_part_dq = pivot_optional("participants_mean_participation_q_delta")
    p_abs_dq = pivot_optional("abstainers_mean_participation_q_delta")
    p_part_dq_proxy = pivot_optional("participants_mean_participation_q_update_proxy")
    p_abs_dq_proxy = pivot_optional("abstainers_mean_participation_q_update_proxy")

    group_resident_count_static: dict[int, float] = {}
    for g in groups:
        vals = p_res[g].to_numpy(dtype=float)
        group_resident_count_static[int(g)] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else float("nan")

    def _group_is_small(g: int) -> bool:
        cnt = group_resident_count_static.get(int(g), float("nan"))
        return bool(np.isfinite(cnt) and cnt < float(_SMALL_GROUP_MIN_RESIDENTS))

    def _group_alpha(g: int, *, normal: float = 0.9, small: float = 0.16) -> float:
        return float(small if _group_is_small(int(g)) else normal)

    has_small_groups = any(_group_is_small(int(g)) for g in groups)

    # Page 1/2: panels 1 + 2 (full-width rows).
    fig1, axes1 = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=True)
    ax1 = np.asarray(axes1).ravel()

    # Panel 1: mu_g vs mu_bar.
    ax1[0].set_title("Participation Learning Base: group mu_g vs mu_bar")
    ax1[0].set_ylabel("delta_rel [%]")
    has_mu = p_mu is not None and np.isfinite(p_mu.to_numpy(dtype=float)).any()
    mu_bar_vals = None
    if has_mu:
        for g in groups:
            y = p_mu[g].to_numpy(dtype=float) * 100.0
            ax1[0].plot(
                steps,
                y,
                color=get_group_color(int(g)),
                linewidth=1.0,
                alpha=_group_alpha(int(g)),
                label=f"g{g}" if not _group_is_small(int(g)) else "_nolegend_",
            )
        if p_mu_bar is not None and np.isfinite(p_mu_bar.to_numpy(dtype=float)).any():
            mu_bar_vals = np.nanmean(p_mu_bar.to_numpy(dtype=float), axis=1)
        else:
            mu_bar_vals = np.nanmean(p_mu.to_numpy(dtype=float), axis=1)
        ax1[0].plot(
            steps,
            mu_bar_vals * 100.0,
            color="black",
            linestyle="--",
            linewidth=1.4,
            alpha=0.95,
            label="mu_bar (cross-group mean)",
        )
        ax1[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=7, ncol=min(5, len(groups) + 1))
    else:
        ax1[0].text(0.5, 0.5, "group_mu/global_mu unavailable", ha="center", va="center")
        ax1[0].set_yticks([])

    # Panel 2: group component with shrink weight context.
    ax1[1].set_title("Group Component: r_g = w_g * (mu_g - mu_bar)")
    ax1[1].set_ylabel("signal component")
    has_component = p_r is not None and np.isfinite(p_r.to_numpy(dtype=float)).any()
    if has_component:
        for g in groups:
            y = p_r[g].to_numpy(dtype=float)
            ax1[1].plot(
                steps,
                y,
                color=get_group_color(int(g)),
                linewidth=1.0,
                alpha=_group_alpha(int(g)),
                label=f"g{g}" if not _group_is_small(int(g)) else "_nolegend_",
            )
    elif has_mu and (p_w is not None) and np.isfinite(p_w.to_numpy(dtype=float)).any():
        mu_mat = p_mu.to_numpy(dtype=float)
        if mu_bar_vals is None:
            mu_bar_vals = np.nanmean(mu_mat, axis=1)
        w_mat = p_w.to_numpy(dtype=float)
        for i, g in enumerate(groups):
            y = w_mat[:, i] * (mu_mat[:, i] - mu_bar_vals)
            ax1[1].plot(
                steps,
                y,
                color=get_group_color(int(g)),
                linewidth=1.0,
                alpha=_group_alpha(int(g)),
                label=f"g{g}" if not _group_is_small(int(g)) else "_nolegend_",
            )
        has_component = True
    if has_component:
        ax1[1].axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.85)
    else:
        ax1[1].text(0.5, 0.5, "group signal component unavailable", ha="center", va="center")
        ax1[1].set_yticks([])

    if p_w is not None and np.isfinite(p_w.to_numpy(dtype=float)).any():
        ax1_rhs = ax1[1].twinx()
        w_mat = p_w.to_numpy(dtype=float)
        w_mean = np.nanmean(w_mat, axis=1)
        w_min = np.nanmin(w_mat, axis=1)
        w_max = np.nanmax(w_mat, axis=1)
        ax1_rhs.fill_between(steps, w_min, w_max, color="gray", alpha=0.14, linewidth=0.0)
        ax1_rhs.plot(steps, w_mean, color="black", linestyle="--", linewidth=1.0, alpha=0.9, label="mean w_g")
        _set_unit_ylim_visible(ax1_rhs)
        ax1_rhs.set_ylabel("shrink weight w_g")
        ax1_rhs.grid(False)

    for a in ax1:
        a.grid(True, alpha=0.25)
        a.set_xlabel("step")
    if has_small_groups:
        fig1.text(
            0.01,
            0.01,
            f"Groups with residents < {_SMALL_GROUP_MIN_RESIDENTS} are plotted transparent.",
            fontsize=7,
            ha="left",
            va="bottom",
            alpha=0.85,
        )
    fig1.suptitle(suptitle + " | Participation Learning Causal Decomposition (1/2)", fontsize=11)
    fig1.tight_layout()
    pdf.savefig(fig1, dpi=140)
    plt.close(fig1)

    # Page 2/2: panels 3 + 4 (full-width rows).
    fig2, axes2 = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=True)
    ax2 = np.asarray(axes2).ravel()

    # Panel 3: signal decomposition by subpopulation.
    ax2[0].set_title("Signal Decomposition by Group")
    ax2[0].set_ylabel("signal")
    has_signal = False
    if p_part_sig is not None and np.isfinite(p_part_sig.to_numpy(dtype=float)).any():
        for g in groups:
            ax2[0].plot(
                steps,
                p_part_sig[g].to_numpy(dtype=float),
                color=get_group_color(int(g)),
                linestyle="-",
                linewidth=1.0,
                alpha=_group_alpha(int(g)),
            )
        has_signal = True
    if p_abs_sig is not None and np.isfinite(p_abs_sig.to_numpy(dtype=float)).any():
        for g in groups:
            ax2[0].plot(
                steps,
                p_abs_sig[g].to_numpy(dtype=float),
                color=get_group_color(int(g)),
                linestyle="--",
                linewidth=1.1,
                alpha=_group_alpha(int(g), normal=0.85, small=0.12),
            )
        has_signal = True
    if p_part_sig_group is not None and np.isfinite(p_part_sig_group.to_numpy(dtype=float)).any():
        for g in groups:
            ax2[0].plot(
                steps,
                p_part_sig_group[g].to_numpy(dtype=float),
                color=get_group_color(int(g)),
                linestyle=":",
                linewidth=1.0,
                alpha=_group_alpha(int(g), normal=0.7, small=0.10),
            )
        has_signal = True
    if p_part_sig_fee is not None and np.isfinite(p_part_sig_fee.to_numpy(dtype=float)).any():
        for g in groups:
            ax2[0].plot(
                steps,
                p_part_sig_fee[g].to_numpy(dtype=float),
                color=get_group_color(int(g)),
                linestyle="-.",
                linewidth=0.95,
                alpha=_group_alpha(int(g), normal=0.8, small=0.11),
            )
        has_signal = True
    if has_signal:
        ax2[0].axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.85)
        group_handles = [
            Line2D([0], [0], color=get_group_color(int(g)), linewidth=1.6, label=f"g{g}")
            for g in groups
            if not _group_is_small(int(g))
        ]
        style_handles = [
            Line2D([0], [0], color="black", linestyle="-", linewidth=1.1, label="participants total signal"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.1, label="abstainers total signal"),
            Line2D([0], [0], color="black", linestyle=":", linewidth=1.1, label="group component"),
            Line2D([0], [0], color="black", linestyle="-.", linewidth=1.1, label="participant fee component"),
        ]
        if group_handles:
            leg_groups = ax2[0].legend(
                handles=group_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.02),
                fontsize=7,
                ncol=min(5, len(group_handles)),
                title="groups",
            )
            ax2[0].add_artist(leg_groups)
        ax2[0].legend(handles=style_handles, loc="upper left", fontsize=7, title="line style")
    else:
        ax2[0].text(0.5, 0.5, "participation signal decomposition unavailable", ha="center", va="center")
        ax2[0].set_yticks([])

    # Panel 4: exact delta-q effect by subpopulation.
    use_exact = (
        p_part_dq is not None
        and p_abs_dq is not None
        and (
            np.isfinite(p_part_dq.to_numpy(dtype=float)).any()
            or np.isfinite(p_abs_dq.to_numpy(dtype=float)).any()
        )
    )
    p_part_effect = p_part_dq if use_exact else p_part_dq_proxy
    p_abs_effect = p_abs_dq if use_exact else p_abs_dq_proxy
    ax2[1].set_title(
        "Participation Update Effect by Group "
        f"({'exact Δq' if use_exact else 'proxy'})"
    )
    ax2[1].set_ylabel("Δq")
    has_effect = (
        p_part_effect is not None
        and p_abs_effect is not None
        and (
            np.isfinite(p_part_effect.to_numpy(dtype=float)).any()
            or np.isfinite(p_abs_effect.to_numpy(dtype=float)).any()
        )
    )
    if has_effect:
        for g in groups:
            color = get_group_color(int(g))
            ax2[1].plot(
                steps,
                p_part_effect[g].to_numpy(dtype=float),
                color=color,
                linestyle="-",
                linewidth=1.0,
                alpha=_group_alpha(int(g)),
                label=f"g{g}" if not _group_is_small(int(g)) else "_nolegend_",
            )
            ax2[1].plot(
                steps,
                p_abs_effect[g].to_numpy(dtype=float),
                color=color,
                linestyle="--",
                linewidth=1.1,
                alpha=_group_alpha(int(g), normal=0.85, small=0.12),
            )
        ax2[1].axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.85)
        if use_exact:
            ax2[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=7, ncol=min(5, len(groups)))
        else:
            ax2[1].text(
                0.01,
                0.03,
                "Proxy mode: participation_alpha * q_push_proxy.",
                transform=ax2[1].transAxes,
                fontsize=7,
                ha="left",
                va="bottom",
                alpha=0.85,
            )
    else:
        ax2[1].text(0.5, 0.5, "participation update effect unavailable", ha="center", va="center")
        ax2[1].set_yticks([])

    for a in ax2:
        a.grid(True, alpha=0.25)
        a.set_xlabel("step")
    if has_small_groups:
        fig2.text(
            0.01,
            0.01,
            f"Groups with residents < {_SMALL_GROUP_MIN_RESIDENTS} are plotted transparent.",
            fontsize=7,
            ha="left",
            va="bottom",
            alpha=0.85,
        )
    fig2.suptitle(suptitle + " | Participation Learning Causal Decomposition (2/2)", fontsize=11)
    fig2.tight_layout()
    pdf.savefig(fig2, dpi=140)
    plt.close(fig2)


def _render_area_group_means_page(
    *,
    pdf: PdfPages,
    area_group_series: pd.DataFrame,
    x: np.ndarray,
    gini_dissatisfaction: np.ndarray,
) -> None:
    groups = sorted(int(v) for v in area_group_series["group_idx"].dropna().unique().tolist())
    steps = np.asarray(sorted(int(v) for v in area_group_series["step"].dropna().unique().tolist()), dtype=float)
    if len(groups) == 0 or steps.size == 0:
        return
    pivot = lambda col: (
        area_group_series.pivot(index="step", columns="group_idx", values=col)
        .reindex(index=steps.astype(int), columns=groups)
        .astype(float)
    )
    p_dissat = pivot("mean_dissatisfaction")

    fig, ax = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=True)
    ax[0].plot(x, gini_dissatisfaction, color="tab:purple", linewidth=1.8)
    ax[0].set_title("Gini Dissatisfaction [0..100]")
    ax[0].set_ylabel("gini")
    _set_percent_ylim_visible(ax[0])
    for g in groups:
        color = get_group_color(int(g))
        ax[1].plot(steps, p_dissat[g].to_numpy(dtype=float), color=color, linewidth=1.8, label=f"g{g}")
    ax[1].set_title("Mean Dissatisfaction by Group")
    ax[1].set_ylabel("dissatisfaction [0..1]")
    _set_unit_ylim_visible(ax[1])
    for a in ax:
        a.grid(True, alpha=0.25)
        a.set_xlabel("step")
    if groups:
        ax[1].legend(loc="best", fontsize=8, ncol=min(5, len(groups)))
    fig.tight_layout()
    pdf.savefig(fig, dpi=140)
    plt.close(fig)


def _prepare_group_assets_share_series(
    *,
    area_group_series: pd.DataFrame,
) -> tuple[np.ndarray, list[int], pd.DataFrame] | None:
    groups = sorted(int(v) for v in area_group_series["group_idx"].dropna().unique().tolist())
    steps = np.asarray(sorted(int(v) for v in area_group_series["step"].dropna().unique().tolist()), dtype=float)
    if len(groups) == 0 or steps.size == 0:
        return None
    pivot = lambda col: (
        area_group_series.pivot(index="step", columns="group_idx", values=col)
        .reindex(index=steps.astype(int), columns=groups)
        .astype(float)
    )
    p_res = pivot("residents")
    p_assets = pivot("mean_assets")
    p_group_assets = p_assets * p_res
    asset_totals = p_group_assets.sum(axis=1).to_numpy(dtype=float)
    p_assets_share = p_group_assets.copy()
    for g in groups:
        vals = p_group_assets[g].to_numpy(dtype=float)
        share = np.zeros_like(vals, dtype=float)
        np.divide(vals, asset_totals, out=share, where=asset_totals > 0.0)
        p_assets_share[g] = share
    return steps, groups, p_assets_share


def _compute_area_weighted_means_from_group_series(*, area_group_series: pd.DataFrame) -> pd.DataFrame | None:
    if area_group_series.empty:
        return None
    required = {"step", "residents", "mean_assets", "mean_dissatisfaction"}
    if not required.issubset(area_group_series.columns):
        return None
    rows: list[dict[str, float]] = []
    for step, block in area_group_series.groupby("step", sort=True):
        w = block["residents"].to_numpy(dtype=float)
        if w.size == 0 or float(np.sum(w)) <= 0.0:
            continue
        assets = block["mean_assets"].to_numpy(dtype=float)
        dissat = block["mean_dissatisfaction"].to_numpy(dtype=float)
        rows.append(
            {
                "step": float(step),
                "mean_assets": float(np.average(assets, weights=w)),
                "mean_dissatisfaction": float(np.average(dissat, weights=w)),
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)

def _render_global_core_metrics_page(
    *,
    pdf: PdfPages,
    global_series: pd.DataFrame,
    meta: dict[str, Any],
    static: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), sharex=True)
    x = global_series["step"].to_numpy(dtype=float)
    ax = axes.ravel()
    ax[0].plot(x, global_series["turnout"].to_numpy(dtype=float), color="tab:blue")
    ax[0].set_title("Turnout [%]")
    _set_percent_ylim_visible(ax[0])
    ax[1].plot(x, global_series["gini_assets"].to_numpy(dtype=float), color="tab:red")
    ax[1].set_title("Gini Assets [0..100]")
    _set_percent_ylim_visible(ax[1])
    ax[2].plot(x, global_series["gini_dissatisfaction"].to_numpy(dtype=float), color="tab:purple")
    ax[2].set_title("Gini Dissatisfaction [0..100]")
    _set_percent_ylim_visible(ax[2])
    ax[3].plot(x, global_series["mean_dissatisfaction"].to_numpy(dtype=float), color="tab:orange")
    ax[3].set_title("Mean Dissatisfaction")
    _set_unit_ylim_visible(ax[3])

    vol_lines = [
        (
            "turnout_volatility",
            step_volatility_l1_normalized(global_series["turnout"].to_numpy(dtype=float), value_range=100.0),
        ),
        (
            "gini_assets_volatility",
            step_volatility_l1_normalized(global_series["gini_assets"].to_numpy(dtype=float), value_range=100.0),
        ),
        (
            "gini_dissatisfaction_volatility",
            step_volatility_l1_normalized(global_series["gini_dissatisfaction"].to_numpy(dtype=float), value_range=100.0),
        ),
        (
            "dist_to_reality_volatility",
            step_volatility_l1_normalized(global_series["dist_to_reality"].to_numpy(dtype=float), value_range=1.0),
        ),
    ]
    vol_text = "Adjacent-step volatility (mean |Δ|)\n" + "\n".join(
        f"{k}: {v:.3f}" if np.isfinite(v) else f"{k}: nan" for k, v in vol_lines
    )
    ax[3].text(
        0.02,
        0.98,
        vol_text,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.5", alpha=0.9),
        transform=ax[3].transAxes,
    )
    for a in ax:
        a.grid(True, alpha=0.25)
        a.set_xlabel("step")
    num_areas = int(static.get("num_areas", 0))
    scope = "Global (= area_0 aggregate; n_areas=1)" if num_areas == 1 else f"Global aggregate (n_areas={num_areas})"
    title = f"Global Core Metrics | run_seed={meta['run']['run_seed']} | rule={meta['run'].get('rule_name')}\n{scope}"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    pdf.savefig(fig, dpi=140)
    plt.close(fig)


def _adjacent_abs_change_series(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    if arr.size == 1:
        return out
    cur = arr[1:]
    prev = arr[:-1]
    mask = np.isfinite(cur) & np.isfinite(prev)
    out[1:] = np.where(mask, np.abs(cur - prev), np.nan)
    return out


def _render_global_step_volatility_page(
    *,
    pdf: PdfPages,
    global_series: pd.DataFrame,
    meta: dict[str, Any],
    static: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(11.69, 8.27), sharex=True)
    x = global_series["step"].to_numpy(dtype=float)
    rows = [
        ("turnout", "Turnout [%]", "tab:blue", "percent"),
        ("gini_assets", "Gini Assets [0..100]", "tab:red", "percent"),
        ("gini_dissatisfaction", "Gini Dissatisfaction [0..100]", "tab:purple", "percent"),
        ("dist_to_reality", "dist_to_reality [0..1]", "tab:green", "unit"),
    ]
    for r, (col, title, color, scale_kind) in enumerate(rows):
        y = global_series[col].to_numpy(dtype=float)
        delta = _adjacent_abs_change_series(y)
        left = axes[r, 0]
        right = axes[r, 1]

        left.plot(x, y, color=color, linewidth=1.8)
        left.set_title(f"{title} level")
        if scale_kind == "percent":
            _set_percent_ylim_visible(left)
        else:
            _set_unit_ylim_visible(left)
        left.set_ylabel("value")
        left.grid(True, alpha=0.25)

        right.plot(x, delta, color=color, linewidth=0.95, alpha=0.25, label="|Δ| per step")
        right.plot(
            x,
            _rolling_mean_nan(delta),
            color=color,
            linewidth=1.6,
            alpha=0.95,
            label=f"rolling mean ({_SMOOTH_WINDOW_STEPS})",
        )
        mean_delta = float(np.nanmean(delta)) if np.isfinite(delta).any() else np.nan
        if np.isfinite(mean_delta):
            right.axhline(
                mean_delta,
                color="black",
                linestyle="--",
                linewidth=1.0,
                alpha=0.85,
                label=f"mean |Δ| = {mean_delta:.3f}",
            )
        finite_delta = delta[np.isfinite(delta)]
        if finite_delta.size > 0:
            ymax = float(np.nanmax(finite_delta))
            right.set_ylim(0.0, max(1e-6, 1.12 * ymax))
        else:
            right.set_ylim(0.0, 1.0)
        right.set_title(f"{title} adjacent-step volatility (mean |Δ|)")
        right.set_ylabel("|Δ|")
        right.grid(True, alpha=0.25)
        right.legend(loc="upper right", fontsize=7)

    for a in axes[-1, :]:
        a.set_xlabel("step")

    num_areas = int(static.get("num_areas", 0))
    scope = "Global (= area_0 aggregate; n_areas=1)" if num_areas == 1 else f"Global aggregate (n_areas={num_areas})"
    fig.suptitle(
        f"Global Step Volatility | run_seed={meta['run']['run_seed']} | rule={meta['run'].get('rule_name')}\n{scope}",
        fontsize=12,
    )
    fig.tight_layout()
    pdf.savefig(fig, dpi=140)
    plt.close(fig)


def _render_global_distance_page(*, pdf: PdfPages, global_series: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), sharex=True)
    x = global_series["step"].to_numpy(dtype=float)
    ax = axes.ravel()
    ax[0].plot(x, global_series["dist_to_reality"].to_numpy(dtype=float), color="tab:green")
    ax[0].set_title("dist_to_reality (weighted)")
    _set_unit_ylim_visible(ax[0])
    ax[1].plot(x, global_series["dist_to_ref_utilitarian"].to_numpy(dtype=float), color="tab:blue", label="utilitarian")
    if "dist_to_ref_nash" in global_series.columns:
        ax[1].plot(x, global_series["dist_to_ref_nash"].to_numpy(dtype=float), color="tab:purple", label="nash")
    ax[1].plot(x, global_series["dist_to_ref_egalitarian"].to_numpy(dtype=float), color="tab:orange", label="egalitarian")
    ax[1].plot(x, global_series["dist_to_ref_rawlsian"].to_numpy(dtype=float), color="tab:red", label="rawlsian")
    ax[1].set_title("dist_to_ref_*")
    _set_unit_ylim_visible(ax[1])
    ax[1].legend(loc="best", fontsize=8)
    ax[2].plot(x, global_series["diversity_first_choice_entropy"].to_numpy(dtype=float), color="tab:brown")
    ax[2].set_title("diversity_first_choice_entropy")
    _set_unit_ylim_visible(ax[2])
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
    _set_unit_ylim_visible(ax_curve)
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
    """Recover resident area->agent ids from typed static overlay artifacts."""
    static_path = run_dir / "static.json"
    if not static_path.exists():
        raise FileNotFoundError(f"Missing static.json for typed overlay lookup: {run_dir}")

    try:
        static = json.loads(static_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as e:
        raise ValueError(f"Failed reading static.json for typed overlays: {static_path}") from e

    artifacts = static.get("artifacts") if isinstance(static.get("artifacts"), dict) else {}
    rel = artifacts.get("cell_agents")
    if not isinstance(rel, str) or rel.strip() == "":
        raise KeyError(f"static.json missing required artifacts.cell_agents: {static_path}")
    cell_agents_path = run_dir / rel
    if not cell_agents_path.exists():
        raise FileNotFoundError(f"Missing typed cell_agents artifact: {cell_agents_path}")

    try:
        df = pd.read_parquet(cell_agents_path)
    except (OSError, ValueError, TypeError) as e:
        raise ValueError(f"Failed reading typed cell_agents artifact: {cell_agents_path}") from e

    if not {"area_id", "agent_id"}.issubset(df.columns):
        raise KeyError(f"{cell_agents_path.name} missing required columns ['area_id', 'agent_id']")

    out: dict[int, set[int]] = {}
    for r in df[["area_id", "agent_id"]].dropna().itertuples(index=False):
        area_id = int(r.area_id)
        agent_id = int(r.agent_id)
        if area_id < 0 or agent_id < 0:
            continue
        if area_id not in out:
            out[area_id] = set()
        out[area_id].add(agent_id)
    return {k: sorted(v) for k, v in sorted(out.items())}


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
    _set_unit_ylim_visible(ax)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["util", "nash", "egal", "rawl"], fontsize=8)
    ax.set_ylabel("share")
    ax.grid(True, axis="y", alpha=0.25)


def _draw_area_personality_group_distribution(
    *,
    ax,
    pg_dist: np.ndarray,
    personality_groups: np.ndarray,
    num_colors: int,
) -> None:
    ax.set_title("Personality Group Dists")
    dist = np.asarray(pg_dist, dtype=float).reshape(-1)
    if dist.size == 0 or not np.isfinite(dist).any():
        ax.axis("off")
        ax.text(0.5, 0.5, "unavailable", ha="center", va="center")
        return
    n_groups = int(dist.size)
    x = np.arange(n_groups, dtype=float)
    # Background: show each group's preference ordering as stacked color stripes.
    if personality_groups.ndim == 2 and personality_groups.shape[0] >= n_groups and int(num_colors) > 0:
        n_slots = int(min(int(num_colors), personality_groups.shape[1]))
        for gi in range(n_groups):
            order = personality_groups[gi].astype(int).tolist()
            for rank, color_id in enumerate(order[:n_slots]):
                y0 = 1.0 - float(rank + 1) / float(n_slots)
                ax.add_patch(
                    plt.Rectangle(
                        (float(gi) - 0.4, y0),
                        0.8,
                        1.0 / float(n_slots),
                        facecolor=_sim_color(color_id),
                        edgecolor="none",
                        alpha=0.26,
                        zorder=0,
                    )
                )
    # Foreground bars: keep black-frame histogram look, add subtle group color fill.
    bars = ax.bar(
        x,
        dist,
        width=0.75,
        facecolor="none",
        edgecolor=[get_group_color(i) for i in range(n_groups)],
        linewidth=1.0,
        zorder=2,
    )
    _set_unit_ylim_visible(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([f"g{i}" for i in range(n_groups)], fontsize=7)
    ax.set_yticks([])
    ax.grid(True, axis="y", alpha=0.2)
    for gi, tick in enumerate(ax.get_xticklabels()):
        c = get_group_color(int(gi))
        r, g, b, _ = to_rgba(c)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        txt = "black" if luminance > 0.55 else "white"
        tick.set_color(txt)
        tick.set_bbox(
            dict(
                facecolor=c,
                edgecolor="none",
                boxstyle="round,pad=0.18,rounding_size=0.08",
                alpha=0.95,
            )
        )
    for b in bars:
        h = float(b.get_height())
        gi = int(round(float(b.get_x() + b.get_width() / 2.0)))
        label_color = get_group_color(max(0, min(n_groups - 1, gi)))
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            min(0.98, h + 0.02),
            f"{100.0 * h:.0f}%",
            ha="center",
            va="bottom",
            fontsize=7,
            color=label_color,
            fontweight="bold",
        )


def _sim_color(color_idx: int) -> str:
    if 0 <= int(color_idx) < len(SIM_COLORS):
        name = SIM_COLORS[int(color_idx)]
        # Matplotlib normalizes both spellings; keep a single one for consistency.
        return "LightGrey" if name == "LightGray" else str(name)
    return "black"


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


def _compute_area_power_direction_orderings(
    *,
    area_group_series: pd.DataFrame,
    personality_groups: np.ndarray,
    num_colors: int,
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    """Static self-regarding counterfactual outcomes per voting rule for this area."""
    if area_group_series.empty or personality_groups.ndim != 2 or num_colors <= 0:
        return []
    if "group_idx" not in area_group_series.columns or "residents" not in area_group_series.columns:
        return []
    step0 = int(area_group_series["step"].min()) if "step" in area_group_series.columns else 1
    block = area_group_series[area_group_series["step"].astype(int) == step0].copy()
    if block.empty:
        return []

    n_groups = int(personality_groups.shape[0])
    residents_by_group = np.zeros(n_groups, dtype=int)
    for _, row in block.iterrows():
        gi = int(row["group_idx"])
        if 0 <= gi < n_groups:
            residents_by_group[gi] = int(max(0, int(row.get("residents", 0))))
    if int(residents_by_group.sum()) <= 0:
        return []

    options = np.asarray(list(itertools.permutations(range(int(num_colors)))), dtype=np.int64)
    dist_func = _summary_ordering_distance_func(meta)
    search_pairs = list(itertools.combinations(range(int(num_colors)), 2))

    pref_rows: list[np.ndarray] = []
    for gi in range(min(n_groups, int(personality_groups.shape[0]))):
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
        return []
    pref_table = np.vstack(pref_rows)

    rule_fns = [majority_rule, approval_voting, utilitarian_rule, borda_rule, random_rule]
    rule_names = ["Majority", "Approval", "Utilitarian", "Borda", "Random"]
    run_seed = int(((meta.get("run", {}) or {}).get("run_seed", 0)) or 0)
    out: list[dict[str, Any]] = []
    for idx, (fn, name) in enumerate(zip(rule_fns, rule_names)):
        rng = np.random.default_rng((run_seed * 1_000_003 + 97 * (idx + 1)) % (2**63 - 1))
        try:
            opt_order = np.asarray(fn(pref_table, rng=rng), dtype=np.int64)
            if opt_order.size <= 0:
                continue
            winning_option_id = int(opt_order[0])
            color_ordering = np.asarray(options[winning_option_id], dtype=np.int64)
            out.append(
                {
                    "rule_idx": int(idx),
                    "rule_name": str(name),
                    "winning_option_id": int(winning_option_id),
                    "color_ordering": color_ordering,
                }
            )
        except (TypeError, ValueError, KeyError, IndexError, RuntimeError) as exc:
            warnings.warn(
                f"Skipping static power-direction baseline for rule '{name}': {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
    return out


def _draw_power_direction_panel(*, ax, power_dirs: list[dict[str, Any]], current_rule_idx: int | None, num_colors: int) -> None:
    ax.set_title("Static Power Directions\n(All Self-Regarding)")
    if not power_dirs or num_colors <= 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "Power baselines unavailable", ha="center", va="center")
        return
    n = len(power_dirs)
    ax.set_xlim(0.0, float(num_colors + 1.9))
    ax.set_ylim(-0.5, float(n - 0.5))
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks(np.arange(n))
    labels = []
    for item in power_dirs:
        nm = str(item.get("rule_name", "rule"))
        ridx = int(item.get("rule_idx", -1))
        labels.append(f"{nm}{' *' if current_rule_idx is not None and ridx == int(current_rule_idx) else ''}")
    ax.set_yticklabels(labels, fontsize=8)
    for row, item in enumerate(power_dirs):
        ordering = np.asarray(item.get("color_ordering", []), dtype=int)
        for rank in range(min(num_colors, ordering.size)):
            c_idx = int(ordering[rank])
            face = _sim_color(c_idx)
            rect = plt.Rectangle(
                (0.9 + rank, row - 0.35),
                0.9,
                0.7,
                facecolor=face,
                edgecolor="black" if int(item.get("rule_idx", -1)) == int(current_rule_idx or -99) else "#666666",
                linewidth=1.8 if int(item.get("rule_idx", -1)) == int(current_rule_idx or -99) else 0.8,
            )
            ax.add_patch(rect)
            r, g, b, _ = to_rgba(face)
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            tcol = "black" if lum > 0.55 else "white"
            ax.text(0.9 + rank + 0.45, row, f"{c_idx}", ha="center", va="center", fontsize=7, color=tcol, fontweight="bold")
            ax.text(0.9 + rank + 0.45, row - 0.46, f"{rank+1}", ha="center", va="top", fontsize=6, color="#333333")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _compute_area_puzzle_power_distances(
    *,
    area_series: pd.DataFrame,
    num_colors: int,
    power_ordering_current_rule: np.ndarray | None,
    meta: dict[str, Any],
) -> dict[str, np.ndarray]:
    xlen = int(len(area_series))
    out = {
        "dist_outcome_power": np.full(xlen, np.nan, dtype=np.float32),
        "dist_puzzle_power": np.full(xlen, np.nan, dtype=np.float32),
        "dist_grid_power": np.full(xlen, np.nan, dtype=np.float32),
    }
    if power_ordering_current_rule is None or num_colors <= 0 or xlen <= 0:
        return out
    power_ord = np.asarray(power_ordering_current_rule, dtype=np.int64).reshape(-1)
    if power_ord.size != int(num_colors):
        return out
    dist_func = _summary_ordering_distance_func(meta)
    search_pairs = list(itertools.combinations(range(int(num_colors)), 2))
    options = np.asarray(list(itertools.permutations(range(int(num_colors)))), dtype=np.int64)

    color_cols = [f"area_color_{i}" for i in range(num_colors) if f"area_color_{i}" in area_series.columns]
    puzzle_cols = [f"puzzle_color_{i}" for i in range(num_colors) if f"puzzle_color_{i}" in area_series.columns]
    has_grid_ids = "grid_ordering_id" in area_series.columns
    has_puzzle_ids = "puzzle_ordering_id" in area_series.columns
    grid_prev = None
    puzzle_prev = None
    tie_rng = np.random.default_rng(int(((meta.get("run", {}) or {}).get("run_seed", 0)) or 0) + 4242)

    win_ids = area_series["winning_option_id"].to_numpy(dtype=int) if "winning_option_id" in area_series.columns else np.full(xlen, -1, dtype=int)
    for i in range(xlen):
        oid = int(win_ids[i])
        if 0 <= oid < int(options.shape[0]):
            out["dist_outcome_power"][i] = np.float32(float(dist_func(np.asarray(options[oid], dtype=np.int64), power_ord, search_pairs)))
        pord = None
        if has_puzzle_ids:
            pid = int(area_series.iloc[i]["puzzle_ordering_id"])
            if 0 <= pid < int(options.shape[0]):
                pord = np.asarray(options[pid], dtype=np.int64)
        elif len(puzzle_cols) == num_colors:
            pvals = area_series.loc[area_series.index[i], puzzle_cols].to_numpy(dtype=float)
            if np.isfinite(pvals).all():
                pord = _summary_ordering_from_distribution_tie_aware(pvals, reference_ordering=puzzle_prev, rng=tie_rng)
        if pord is not None:
            out["dist_puzzle_power"][i] = np.float32(float(dist_func(pord, power_ord, search_pairs)))
            puzzle_prev = pord
        gord = None
        if has_grid_ids:
            gid = int(area_series.iloc[i]["grid_ordering_id"])
            if 0 <= gid < int(options.shape[0]):
                gord = np.asarray(options[gid], dtype=np.int64)
        elif len(color_cols) == num_colors:
            gvals = area_series.loc[area_series.index[i], color_cols].to_numpy(dtype=float)
            if np.isfinite(gvals).all():
                gord = _summary_ordering_from_distribution_tie_aware(gvals, reference_ordering=grid_prev, rng=tie_rng)
        if gord is not None:
            out["dist_grid_power"][i] = np.float32(float(dist_func(gord, power_ord, search_pairs)))
            grid_prev = gord
    return out


def _compute_area_puzzle_anti_monopoly_gate_metrics(
    *,
    area_series: pd.DataFrame,
    dist_decomp: dict[str, np.ndarray],
) -> dict[str, Any]:
    xlen = int(len(area_series))
    thresholds = {
        "conflict_min_dist": float(DEFAULT_SCORING_THRESHOLDS["puzzle_conflict_min_dist"]),
        "min_conflict_share": float(DEFAULT_SCORING_THRESHOLDS["min_puzzle_conflict_step_share_for_gate"]),
        "max_dominance_share_conflict": float(DEFAULT_SCORING_THRESHOLDS["max_puzzle_dominance_share_conflict"]),
        "min_recovery_share_conflict": float(DEFAULT_SCORING_THRESHOLDS["min_power_recovery_share_conflict"]),
    }
    out: dict[str, Any] = {
        **thresholds,
        "d_out_puz": np.full(xlen, np.nan, dtype=np.float32),
        "d_out_pow": np.full(xlen, np.nan, dtype=np.float32),
        "d_puz_pow": np.full(xlen, np.nan, dtype=np.float32),
        "margin": np.full(xlen, np.nan, dtype=np.float32),
        "conflict_mask": np.zeros(xlen, dtype=bool),
        "puzzle_max_share": np.full(xlen, np.nan, dtype=np.float32),
        "puzzle_entropy_norm": np.full(xlen, np.nan, dtype=np.float32),
        "puzzle_conflict_step_share": np.nan,
        "puzzle_dominance_share_conflict": np.nan,
        "power_recovery_share_conflict": np.nan,
        "puzzle_power_margin_mean_conflict": np.nan,
        "gate_puzzle_anti_monopoly": True,
        "puzzle_metric_available": False,
        "gate_conflict_eligible": False,
        "valid_rows": 0,
        "conflict_rows": 0,
    }
    if xlen <= 0:
        return out

    d_out_puz = area_series.get("puzzle_distance", pd.Series(np.nan, index=area_series.index)).to_numpy(dtype=float)
    d_out_pow = np.asarray(dist_decomp.get("dist_outcome_power", np.full(xlen, np.nan, dtype=np.float32)), dtype=float)
    d_puz_pow = np.asarray(dist_decomp.get("dist_puzzle_power", np.full(xlen, np.nan, dtype=np.float32)), dtype=float)
    if d_out_pow.size != xlen:
        d_out_pow = np.full(xlen, np.nan, dtype=float)
    if d_puz_pow.size != xlen:
        d_puz_pow = np.full(xlen, np.nan, dtype=float)

    margin = d_out_pow - d_out_puz
    valid = np.isfinite(d_out_puz) & np.isfinite(d_out_pow) & np.isfinite(d_puz_pow)
    conflict_mask = valid & (d_puz_pow >= float(thresholds["conflict_min_dist"]))

    out["d_out_puz"] = d_out_puz.astype(np.float32)
    out["d_out_pow"] = d_out_pow.astype(np.float32)
    out["d_puz_pow"] = d_puz_pow.astype(np.float32)
    out["margin"] = margin.astype(np.float32)
    out["conflict_mask"] = conflict_mask.astype(bool)
    out["valid_rows"] = int(np.count_nonzero(valid))
    out["conflict_rows"] = int(np.count_nonzero(conflict_mask))

    if int(np.count_nonzero(valid)) > 0:
        out["puzzle_conflict_step_share"] = float(np.mean(conflict_mask[valid]))
    if int(np.count_nonzero(conflict_mask)) > 0:
        conflict_margins = margin[conflict_mask]
        conflict_margins = conflict_margins[np.isfinite(conflict_margins)]
        if conflict_margins.size > 0:
            out["puzzle_dominance_share_conflict"] = float(np.mean(conflict_margins > 0.0))
            out["power_recovery_share_conflict"] = float(np.mean(conflict_margins < 0.0))
            out["puzzle_power_margin_mean_conflict"] = float(np.mean(conflict_margins))

    metrics_available = (
        np.isfinite(float(out["puzzle_conflict_step_share"]))
        and np.isfinite(float(out["puzzle_dominance_share_conflict"]))
        and np.isfinite(float(out["power_recovery_share_conflict"]))
    )
    enough_conflict = (
        metrics_available
        and float(out["puzzle_conflict_step_share"]) >= float(thresholds["min_conflict_share"])
    )
    anti_monopoly_ok = (
        metrics_available
        and float(out["puzzle_dominance_share_conflict"]) <= float(thresholds["max_dominance_share_conflict"])
        and float(out["power_recovery_share_conflict"]) >= float(thresholds["min_recovery_share_conflict"])
    )
    out["puzzle_metric_available"] = bool(metrics_available)
    out["gate_conflict_eligible"] = bool(enough_conflict)
    out["gate_puzzle_anti_monopoly"] = bool(anti_monopoly_ok if enough_conflict else True)

    puzzle_cols = sorted([c for c in area_series.columns if c.startswith("puzzle_color_")], key=lambda c: int(c.split("_")[-1]))
    if puzzle_cols:
        pdata = area_series[puzzle_cols].to_numpy(dtype=float)
        n_cols = int(len(puzzle_cols))
        max_share = np.full(xlen, np.nan, dtype=np.float32)
        entropy_norm = np.full(xlen, np.nan, dtype=np.float32)
        for i in range(xlen):
            row = pdata[i]
            if not np.isfinite(row).all():
                continue
            s = float(np.sum(row))
            if s <= 0.0:
                continue
            p = row / s
            max_share[i] = np.float32(float(np.max(p)))
            if n_cols > 1:
                h = float(-np.sum(p * np.log(p + 1e-15)))
                entropy_norm[i] = np.float32(h / float(np.log(float(n_cols))))
        out["puzzle_max_share"] = max_share
        out["puzzle_entropy_norm"] = entropy_norm
    return out


def _render_area_puzzle_gate_page(
    *,
    pdf: PdfPages,
    area_series: pd.DataFrame,
    dist_decomp: dict[str, np.ndarray],
    suptitle: str,
    include_overview: bool = True,
    include_decomposition: bool = True,
) -> None:
    metrics = _compute_area_puzzle_anti_monopoly_gate_metrics(
        area_series=area_series,
        dist_decomp=dist_decomp,
    )
    x = area_series["step"].to_numpy(dtype=float)

    status_ok = bool(metrics["gate_puzzle_anti_monopoly"])
    status_txt = "PASS" if status_ok else "FAIL"
    status_color = "tab:green" if status_ok else "tab:red"
    conf_eligible = bool(metrics["gate_conflict_eligible"])
    metric_ready = bool(metrics["puzzle_metric_available"])

    d_out_puz = np.asarray(metrics["d_out_puz"], dtype=float)
    d_out_pow = np.asarray(metrics["d_out_pow"], dtype=float)
    d_puz_pow = np.asarray(metrics["d_puz_pow"], dtype=float)
    d_grid_pow = np.asarray(dist_decomp.get("dist_grid_power", np.full_like(d_puz_pow, np.nan, dtype=float)), dtype=float)
    conflict_mask = np.asarray(metrics["conflict_mask"], dtype=bool)
    margin = np.asarray(metrics["margin"], dtype=float)

    # Page A: gate status + puzzle concentration signals.
    if include_overview:
        figa, axa = plt.subplots(1, 2, figsize=(11.69, 8.27))
        axa = np.asarray(axa).ravel()

        axa[0].axis("off")
        axa[0].text(
            0.02,
            0.98,
            "Puzzle Anti-Monopoly Gate",
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
        )
        axa[0].text(
            0.02,
            0.80,
            status_txt,
            ha="left",
            va="top",
            fontsize=24,
            color=status_color,
            fontweight="bold",
        )
        lines = [
            f"metrics_available: {metric_ready}",
            f"conflict_eligible: {conf_eligible}",
            f"conflict_share: {float(metrics['puzzle_conflict_step_share']):.3f} (>= {float(metrics['min_conflict_share']):.3f})",
            f"dominance_share_conflict: {float(metrics['puzzle_dominance_share_conflict']):.3f} (<= {float(metrics['max_dominance_share_conflict']):.3f})",
            f"recovery_share_conflict: {float(metrics['power_recovery_share_conflict']):.3f} (>= {float(metrics['min_recovery_share_conflict']):.3f})",
            f"margin_mean_conflict: {float(metrics['puzzle_power_margin_mean_conflict']):.3f}",
            f"valid_rows: {int(metrics['valid_rows'])}",
            f"conflict_rows: {int(metrics['conflict_rows'])}",
        ]
        axa[0].text(0.02, 0.60, "\n".join(lines), ha="left", va="top", fontsize=9)

        max_share = np.asarray(metrics["puzzle_max_share"], dtype=float)
        entropy = np.asarray(metrics["puzzle_entropy_norm"], dtype=float)
        if np.isfinite(max_share).any():
            axa[1].plot(x, max_share, color="tab:red", linewidth=1.4, label="max puzzle color share")
        if np.isfinite(entropy).any():
            ax1b = axa[1].twinx()
            ax1b.plot(x, entropy, color="tab:blue", linestyle="--", linewidth=1.2, label="puzzle entropy (norm)")
            ax1b.set_ylim(-0.02, 1.02)
            ax1b.set_ylabel("entropy [0..1]", color="tab:blue")
            ax1b.tick_params(axis="y", colors="tab:blue")
            h1, l1 = axa[1].get_legend_handles_labels()
            h2, l2 = ax1b.get_legend_handles_labels()
            if h1 or h2:
                axa[1].legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
        elif len(axa[1].lines) > 0:
            axa[1].legend(loc="best", fontsize=8)
        axa[1].set_title("Puzzle Concentration Signals")
        axa[1].set_ylabel("max share [0..1]", color="tab:red")
        axa[1].tick_params(axis="y", colors="tab:red")
        _set_unit_ylim_visible(axa[1])
        axa[1].grid(True, alpha=0.25)
        axa[1].set_xlabel("step")

        figa.suptitle(suptitle + " | Puzzle Anti-Monopoly Gate (Overview)", fontsize=11)
        figa.tight_layout()
        pdf.savefig(figa, dpi=140)
        plt.close(figa)

    # Page B: split decomposition (3 panels) to avoid overload.
    if include_decomposition:
        figb, axb = plt.subplots(3, 1, figsize=(11.69, 8.27), sharex=True)
        axb = np.asarray(axb).ravel()

        if np.isfinite(d_out_puz).any() or np.isfinite(d_out_pow).any():
            if np.isfinite(d_out_puz).any():
                axb[0].plot(x, d_out_puz, color="black", linestyle="--", linewidth=1.3, label="outcome↔puzzle")
            if np.isfinite(d_out_pow).any():
                axb[0].plot(x, d_out_pow, color="tab:red", linewidth=1.2, label="outcome↔power")
            if conflict_mask.any():
                axb[0].fill_between(x, 0.0, 1.0, where=conflict_mask, color="tab:blue", alpha=0.06, step="mid")
            if len(axb[0].lines) > 0:
                axb[0].legend(loc="best", fontsize=8)
        else:
            axb[0].text(0.5, 0.5, "Outcome distance series unavailable", ha="center", va="center")
            axb[0].set_yticks([])
        axb[0].set_title("Puzzle / Power Distance Decomposition A: Outcome↔Puzzle and Outcome↔Power")
        axb[0].set_ylabel("distance [0..1]")
        _set_unit_ylim_visible(axb[0])

        has_pairwise = False
        if np.isfinite(d_puz_pow).any():
            has_pairwise = True
            axb[1].plot(x, d_puz_pow, color="tab:blue", linewidth=1.2, label="puzzle↔power")
        if np.isfinite(d_grid_pow).any():
            has_pairwise = True
            axb[1].plot(x, d_grid_pow, color="tab:orange", linewidth=1.2, linestyle=":", label="grid↔power")
        axb[1].axhline(
            float(metrics["conflict_min_dist"]),
            color="tab:blue",
            linestyle=":",
            linewidth=1.15,
            label="conflict threshold",
        )
        if conflict_mask.any():
            axb[1].fill_between(x, 0.0, 1.0, where=conflict_mask, color="tab:blue", alpha=0.06, step="mid")
        if has_pairwise:
            axb[1].legend(loc="best", fontsize=8)
        else:
            axb[1].text(0.5, 0.5, "Pairwise puzzle/power diagnostics unavailable", ha="center", va="center")
            axb[1].set_yticks([])
        axb[1].set_title("Puzzle / Power Distance Decomposition B: Puzzle↔Power and Grid↔Power")
        axb[1].set_ylabel("distance [0..1]")
        _set_unit_ylim_visible(axb[1])

        if np.isfinite(margin).any():
            axb[2].plot(x, margin, color="purple", linewidth=1.15, label="margin = outcome↔power - outcome↔puzzle")
            axb[2].axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.85)
            if conflict_mask.any():
                axb[2].scatter(
                    x[conflict_mask],
                    margin[conflict_mask],
                    s=9,
                    color="purple",
                    alpha=0.65,
                    label="conflict steps",
                )
            if len(axb[2].lines) > 0:
                axb[2].legend(loc="best", fontsize=8)
        else:
            axb[2].text(0.5, 0.5, "Margin unavailable", ha="center", va="center")
            axb[2].set_yticks([])
        axb[2].set_title("Puzzle / Power Distance Decomposition C: Margin Around Puzzle vs Power")
        axb[2].set_ylabel("margin [-1..1]")
        axb[2].set_ylim(-1.02, 1.02)

        for a in axb:
            a.grid(True, alpha=0.25)
            a.set_xlabel("step")

        figb.suptitle(suptitle + " | Puzzle Anti-Monopoly Gate (Distance Decomposition)", fontsize=11)
        figb.tight_layout()
        pdf.savefig(figb, dpi=140)
        plt.close(figb)



def _compute_group_puzzle_opportunity_distances(
    *,
    area_series: pd.DataFrame,
    personality_groups: np.ndarray,
    num_colors: int,
    meta: dict[str, Any],
) -> pd.DataFrame:
    """Per-step distance between puzzle ordering and each personality-group ordering."""
    if num_colors <= 0 or personality_groups.ndim != 2:
        return pd.DataFrame()
    puzzle_cols = [f"puzzle_color_{i}" for i in range(num_colors) if f"puzzle_color_{i}" in area_series.columns]
    has_puzzle_ids = "puzzle_ordering_id" in area_series.columns
    if (not has_puzzle_ids and len(puzzle_cols) != num_colors) or area_series.empty:
        return pd.DataFrame()
    dist_func = _summary_ordering_distance_func(meta)
    search_pairs = list(itertools.combinations(range(int(num_colors)), 2))
    x = area_series["step"].to_numpy(dtype=int)
    out = pd.DataFrame({"step": x.astype(np.int32)})
    prev_ord = None
    tie_rng = np.random.default_rng(int(((meta.get("run", {}) or {}).get("run_seed", 0)) or 0) + 7171)
    options = np.asarray(list(itertools.permutations(range(int(num_colors)))), dtype=int)
    puzzle_orders: list[np.ndarray | None] = []
    for _, r in area_series.iterrows():
        pord = None
        if has_puzzle_ids:
            pid = int(r.get("puzzle_ordering_id", -1))
            if 0 <= pid < int(options.shape[0]):
                pord = np.asarray(options[pid], dtype=np.int64)
        elif len(puzzle_cols) == num_colors:
            vals = r[puzzle_cols].to_numpy(dtype=float)
            if np.isfinite(vals).all():
                pord = _summary_ordering_from_distribution_tie_aware(vals, reference_ordering=prev_ord, rng=tie_rng)
        if pord is not None:
            prev_ord = pord
            puzzle_orders.append(pord)
        else:
            puzzle_orders.append(None)
    n_groups = int(personality_groups.shape[0])
    for gi in range(n_groups):
        target = np.asarray(personality_groups[gi][:num_colors], dtype=np.int64)
        vals = np.full(len(puzzle_orders), np.nan, dtype=np.float32)
        for i, pord in enumerate(puzzle_orders):
            if pord is None:
                continue
            vals[i] = np.float32(float(dist_func(np.asarray(pord, dtype=np.int64), target, search_pairs)))
        out[f"group_{gi}_puzzle_opp_dist"] = vals
    return out


def _build_elected_ordering_background_image(
    *,
    winning_option_ids: np.ndarray,
    num_colors: int,
    alpha: float = 0.28,
) -> np.ndarray | None:
    """Build RGBA image for elected ordering background in dist_to_reality plots."""
    ids = np.asarray(winning_option_ids, dtype=int).reshape(-1)
    n_steps = int(ids.size)
    if n_steps <= 0 or num_colors <= 0:
        return None

    options = np.asarray(list(itertools.permutations(range(int(num_colors)))), dtype=int)
    rgba = np.zeros((int(num_colors), n_steps, 4), dtype=np.float32)

    for t, oid in enumerate(ids.tolist()):
        if oid < 0 or oid >= int(options.shape[0]):
            # transparent for missing/invalid winner rows
            rgba[:, t, :] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            continue
        ordering = options[int(oid)]
        for rank in range(int(num_colors)):
            c_idx = int(ordering[rank])
            r, g, b, _ = to_rgba(_sim_color(c_idx))
            rgba[rank, t, :] = np.array([r, g, b, float(alpha)], dtype=np.float32)
    return rgba


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
    for gi, tick in enumerate(ax.get_yticklabels()):
        c = get_group_color(int(gi))
        r, g, b, _ = to_rgba(c)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        txt = "black" if luminance > 0.55 else "white"
        tick.set_color(txt)
        tick.set_bbox(
            dict(
                facecolor=c,
                edgecolor="none",
                boxstyle="round,pad=0.20,rounding_size=0.08",
                alpha=0.95,
            )
        )
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
