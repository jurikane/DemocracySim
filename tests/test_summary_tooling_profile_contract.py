from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_headless import run_once
from src.analysis.summary_tooling import (
    SUMMARY_PROFILE_DEBUG_DOE_COMPACT,
    SUMMARY_PROFILE_FULL,
    generate_run_summary_batch2,
    _resolve_summary_render_profile,
)
from src.config.loader import load_config


def _make_run(tmp_path: Path) -> Path:
    cfg = load_config("toy.yaml").model_copy(deep=True)
    cfg.simulation.num_steps = 8
    cfg.simulation.base_seed = 17401
    out = tmp_path / "run"
    run_once(run_id=0, cfg=cfg, out_dir=out)
    return out


def test_resolve_debug_profile_single_area_disables_duplicate_area_core() -> None:
    profile = _resolve_summary_render_profile(
        profile=SUMMARY_PROFILE_DEBUG_DOE_COMPACT,
        num_areas=1,
    )
    assert profile.global_core_metrics
    assert profile.global_distance_metrics
    assert not profile.global_colors_and_grids
    assert not profile.area_core_page
    assert profile.area_puzzle_page
    assert profile.area_vote_mode_alignment_page
    assert profile.area_group_opportunity_page
    assert profile.area_puzzle_gate_page
    assert not profile.area_group_diagnostics_pages
    assert not profile.area_assets_page
    assert not profile.area_group_means_page
    assert not profile.area_dist_to_ref_page


def test_resolve_debug_profile_multi_area_keeps_area_core() -> None:
    profile = _resolve_summary_render_profile(
        profile=SUMMARY_PROFILE_DEBUG_DOE_COMPACT,
        num_areas=2,
    )
    assert profile.area_core_page


def test_batch2_accepts_debug_profile_and_writes_pdfs(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    artifacts = generate_run_summary_batch2(
        run_dir=run_dir,
        profile=SUMMARY_PROFILE_DEBUG_DOE_COMPACT,
    )
    assert artifacts.global_summary_pdf is not None
    assert artifacts.global_summary_pdf.exists()
    assert any(artifacts.out_dir.glob("area_*.pdf"))


def test_batch2_accepts_external_out_dir_without_needing_local_config_copy(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    out_dir = tmp_path / "review_bundle_like" / "packet" / "summary"
    artifacts = generate_run_summary_batch2(
        run_dir=run_dir,
        out_dir=out_dir,
        profile=SUMMARY_PROFILE_DEBUG_DOE_COMPACT,
    )
    assert artifacts.out_dir == out_dir
    assert artifacts.global_summary_pdf is not None
    assert artifacts.global_summary_pdf.exists()
    assert any(out_dir.glob("area_*.pdf"))


def test_invalid_profile_rejected() -> None:
    with pytest.raises(ValueError):
        _resolve_summary_render_profile(profile="invalid", num_areas=1)


def test_full_profile_keeps_full_render_flags() -> None:
    profile = _resolve_summary_render_profile(
        profile=SUMMARY_PROFILE_FULL,
        num_areas=1,
    )
    assert profile.global_colors_and_grids
    assert profile.global_static_overview
    assert profile.global_per_area_group_distribution
    assert not profile.area_puzzle_gate_page
    assert profile.area_group_diagnostics_pages
    assert profile.area_dist_to_ref_page
