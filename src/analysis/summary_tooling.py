from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import re

import pandas as pd

from src.analysis import summary_references as _summary_references
from src.analysis.reference_benchmarks import (
    egalitarian_refs_mean_plus_lambda_gini,
    nash_ref_kl,
    rawlsian_ref_minimax_l2sq,
    utilitarian_ref_l2sq,
)
from src.analysis.summary_io import (
    _load_altruism_alpha_for_run,
    _load_altruism_learning_for_run,
    _load_area_agent_ids_from_static_overlays,
    _load_participation_alpha_for_run,
    _load_participation_signal_group_shrink_k_for_run,
    _load_participation_signal_mode_for_run,
    _load_required_run_meta_static,
)
from src.analysis.summary_render_area import (
    _build_elected_ordering_background_image,
    _compute_area_power_direction_orderings,
    _draw_area_personality_group_distribution,
    _render_area_detail_pdfs,
    _sim_color,
)
from src.analysis.summary_render_global import _render_combined_global_summary_pdf
from src.analysis.summary_series import _build_area_group_series, _build_area_series, _build_global_series
from src.analysis.summary_stats import _build_summary_stats


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


def _sync_reference_function_bindings() -> None:
    """Keep reference payload module bound to this module-level benchmark callables."""
    _summary_references.utilitarian_ref_l2sq = utilitarian_ref_l2sq
    _summary_references.nash_ref_kl = nash_ref_kl
    _summary_references.egalitarian_refs_mean_plus_lambda_gini = egalitarian_refs_mean_plus_lambda_gini
    _summary_references.rawlsian_ref_minimax_l2sq = rawlsian_ref_minimax_l2sq


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
    _sync_reference_function_bindings()
    refs = _summary_references._load_or_compute_reference_payload(
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
        quality_target_mode=str(meta["run"].get("quality_target_mode", "reality")),
    )
    area_series = _build_area_series(
        area_steps=area_steps,
        votes=votes,
        agents=agents,
        area_agent_ids=area_agent_ids,
        num_colors=num_colors,
        refs_by_area=refs["areas"],
        quality_target_mode=str(meta["run"].get("quality_target_mode", "reality")),
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
    _sync_reference_function_bindings()
    refs_payload = _summary_references._load_or_compute_reference_payload(
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


def _validate_summary_mode(mode: str) -> None:
    if str(mode) not in _SUMMARY_MODES:
        raise ValueError(f"Unsupported summary mode '{mode}'. Expected one of {_SUMMARY_MODES}.")


def _validate_summary_profile(profile: str) -> None:
    if str(profile) not in _SUMMARY_PROFILES:
        raise ValueError(f"Unsupported summary profile '{profile}'. Expected one of {_SUMMARY_PROFILES}.")


def _resolve_summary_render_profile(*, profile: str, num_areas: int) -> _SummaryRenderProfile:
    _validate_summary_profile(profile)
    if profile == SUMMARY_PROFILE_DEBUG_DOE_COMPACT:
        if int(num_areas) <= 1:
            return _SummaryRenderProfile(
                name=SUMMARY_PROFILE_DEBUG_DOE_COMPACT,
                global_colors_and_grids=False,
                global_static_overview=False,
                global_per_area_group_distribution=False,
                global_core_metrics=True,
                global_step_volatility_page=True,
                global_distance_metrics=True,
                area_core_page=True,
                area_puzzle_page=True,
                area_vote_mode_alignment_page=False,
                area_group_opportunity_page=True,
                area_puzzle_gate_page=True,
                area_group_diagnostics_pages=True,
                area_learning_causal_page=True,
                area_assets_page=False,
                area_group_means_page=False,
                area_dist_to_ref_page=False,
            )
        return _SummaryRenderProfile(
            name=SUMMARY_PROFILE_DEBUG_DOE_COMPACT,
            global_colors_and_grids=False,
            global_static_overview=True,
            global_per_area_group_distribution=False,
            global_core_metrics=True,
            global_step_volatility_page=True,
            global_distance_metrics=True,
            area_core_page=True,
            area_puzzle_page=True,
            area_vote_mode_alignment_page=False,
            area_group_opportunity_page=True,
            area_puzzle_gate_page=True,
            area_group_diagnostics_pages=True,
            area_learning_causal_page=True,
            area_assets_page=False,
            area_group_means_page=False,
            area_dist_to_ref_page=False,
        )
    if profile == SUMMARY_PROFILE_THESIS_CORE:
        return _SummaryRenderProfile(
            name=SUMMARY_PROFILE_THESIS_CORE,
            global_colors_and_grids=False,
            global_static_overview=True,
            global_per_area_group_distribution=False,
            global_core_metrics=True,
            global_step_volatility_page=True,
            global_distance_metrics=True,
            area_core_page=True,
            area_puzzle_page=True,
            area_vote_mode_alignment_page=False,
            area_group_opportunity_page=True,
            area_puzzle_gate_page=True,
            area_group_diagnostics_pages=True,
            area_learning_causal_page=True,
            area_assets_page=False,
            area_group_means_page=True,
            area_dist_to_ref_page=False,
        )
    return _SummaryRenderProfile(
        name=SUMMARY_PROFILE_FULL,
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
