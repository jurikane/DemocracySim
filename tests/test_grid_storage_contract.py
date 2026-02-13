from __future__ import annotations

import json
from pathlib import Path

from scripts.run_headless import run_once
from src.config.loader import load_config


def _run(
    tmp_path: Path,
    *,
    num_steps: int,
    store_grid: bool,
    grid_interval: int,
    seed: int,
    label: str,
) -> Path:
    cfg = load_config("toy.yaml").model_copy(deep=True)
    cfg.simulation.num_steps = int(num_steps)
    cfg.simulation.store_grid = bool(store_grid)
    cfg.simulation.grid_interval = int(grid_interval)
    cfg.simulation.base_seed = int(seed)
    out_dir = tmp_path / label
    run_once(run_id=0, cfg=cfg, out_dir=out_dir)
    return out_dir


def _grid_indices(run_dir: Path) -> list[int]:
    grids_dir = run_dir / "grids"
    if not grids_dir.exists():
        return []
    idx: list[int] = []
    for p in sorted(grids_dir.glob("grid_*.npy")):
        s = p.stem.rsplit("_", 1)[-1]
        if s.isdigit():
            idx.append(int(s))
    return idx


def test_store_grid_oracle_false_writes_no_grid_artifacts(tmp_path: Path) -> None:
    out = _run(
        tmp_path,
        num_steps=4,
        store_grid=False,
        grid_interval=1,
        seed=5101,
        label="no_grid",
    )
    assert not (out / "grids").exists()
    # Core schema-v2 parquet artifacts still must exist.
    for name in ("steps.parquet", "area_steps.parquet", "agents.parquet", "votes.parquet"):
        assert (out / name).exists(), name


def test_grid_interval_metamorphic_changes_written_step_indices(tmp_path: Path) -> None:
    # num_steps=6 => recorded steps are 1..6. With interval=2, write at step indices 1,3,5.
    out_i1 = _run(
        tmp_path,
        num_steps=6,
        store_grid=True,
        grid_interval=1,
        seed=5102,
        label="i1",
    )
    out_i2 = _run(
        tmp_path,
        num_steps=6,
        store_grid=True,
        grid_interval=2,
        seed=5102,
        label="i2",
    )

    idx1 = _grid_indices(out_i1)
    idx2 = _grid_indices(out_i2)
    assert idx1 == [0, 1, 2, 3, 4, 5, 6]
    assert idx2 == [0, 1, 3, 5]
    assert len(idx2) < len(idx1)


def test_grid_filename_and_indexing_integration_expectations(tmp_path: Path) -> None:
    out = _run(
        tmp_path,
        num_steps=12,  # pad = len("12") = 2
        store_grid=True,
        grid_interval=3,
        seed=5103,
        label="n12_i3",
    )

    static = json.loads((out / "static.json").read_text(encoding="utf-8"))
    assert static["step_indexing"]["grid_file"] == "grid_%02d.npy"

    grids = sorted((out / "grids").glob("grid_*.npy"))
    names = [p.name for p in grids]
    # Initial pre-election snapshot must exist.
    assert "grid_00.npy" in names
    # Per-step snapshots for interval=3: steps 1,4,7,10.
    for expected in ("grid_01.npy", "grid_04.npy", "grid_07.npy", "grid_10.npy"):
        assert expected in names
    # Every index must be zero-padded to width 2.
    for name in names:
        suffix = name.removeprefix("grid_").removesuffix(".npy")
        assert len(suffix) == 2
