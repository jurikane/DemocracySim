from __future__ import annotations

from pathlib import Path

from scripts.run_headless import run_once
from src.analysis.summary_tooling import generate_run_summary_batch2
from src.config.loader import load_config


def _make_run(tmp_path: Path) -> Path:
    cfg = load_config("toy.yaml").model_copy(deep=True)
    cfg.simulation.num_steps = 8
    cfg.simulation.base_seed = 17601
    cfg.simulation.store_grid = False
    out = tmp_path / "run_0"
    run_once(run_id=0, cfg=cfg, out_dir=out)
    return out


def test_batch2_writes_area_detail_pdfs_for_all_areas(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    artifacts = generate_run_summary_batch2(run_dir=run_dir, mode="full", use_cache=True)
    assert artifacts.area_group_series_csv is not None
    assert artifacts.area_group_series_csv.exists()
    assert artifacts.area_group_series_csv.stat().st_size > 0

    static = (run_dir / "static.json").read_text(encoding="utf-8")
    import json
    num_areas = int(json.loads(static)["num_areas"])

    for area_id in range(num_areas):
        p = artifacts.out_dir / f"area_{area_id}.pdf"
        assert p.exists(), f"missing area detail pdf: {p.name}"
        assert p.stat().st_size > 0, f"empty area detail pdf: {p.name}"
