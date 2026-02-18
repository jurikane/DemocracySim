from __future__ import annotations

from pathlib import Path

from scripts.run_headless import run_once
from src.analysis.summary_tooling import generate_run_summary_batch2
from src.config.loader import load_config


def _make_run(tmp_path: Path, *, store_grid: bool) -> Path:
    cfg = load_config("toy.yaml").model_copy(deep=True)
    cfg.simulation.num_steps = 8
    cfg.simulation.base_seed = 17201
    cfg.simulation.store_grid = bool(store_grid)
    out = tmp_path / ("run_grid" if store_grid else "run_no_grid")
    run_once(run_id=0, cfg=cfg, out_dir=out)
    return out


def test_batch2_writes_static_and_global_pdf_artifacts(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, store_grid=True)
    artifacts = generate_run_summary_batch2(run_dir=run_dir)

    assert artifacts.global_summary_pdf is not None
    assert artifacts.static_overview_pdf is None
    assert artifacts.global_summary_pdf.exists()
    assert artifacts.global_summary_pdf.stat().st_size > 0
    assert artifacts.global_summary_pdf.name.startswith("global_summary_")
    assert "_seed" in artifacts.global_summary_pdf.name

    # Batch-1 sidecars are still present and generated in the same run.
    assert artifacts.global_series_csv.exists()
    assert artifacts.area_series_csv.exists()
    assert artifacts.summary_stats_json.exists()


def test_batch2_global_pdf_generation_is_stable_without_grids(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, store_grid=False)
    artifacts = generate_run_summary_batch2(run_dir=run_dir)

    assert artifacts.global_summary_pdf is not None
    assert artifacts.global_summary_pdf.exists()
    assert artifacts.global_summary_pdf.stat().st_size > 0
