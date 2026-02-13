from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from scripts.run_headless import batch_run
from src.config.loader import load_config


def _write_cfg(path: Path, *, base_seed: int, runs: int, out_dir: Path) -> None:
    cfg = load_config("toy.yaml").model_copy(deep=True)
    cfg.simulation.base_seed = base_seed
    cfg.simulation.runs = runs
    cfg.simulation.num_steps = 1
    cfg.simulation.store_grid = False
    cfg.simulation.grid_interval = 1
    payload = cfg.model_dump(mode="json")
    payload["output"] = {"directory": str(out_dir)}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _latest_run_root(base_out: Path) -> Path:
    roots = sorted([p for p in base_out.iterdir() if p.is_dir()])
    assert roots, f"No run root created under {base_out}"
    return roots[-1]


def _meta_seed(run_dir: Path) -> int:
    payload = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
    return int(payload["run"]["run_seed"])


def test_base_seed_oracle_batch_run_assigns_run_seeds_linearly(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    out_base = tmp_path / "out"
    _write_cfg(cfg_path, base_seed=2000, runs=3, out_dir=out_base)

    batch_run(str(cfg_path))
    run_root = _latest_run_root(out_base)

    assert _meta_seed(run_root / "run_0") == 2000
    assert _meta_seed(run_root / "run_1") == 2001
    assert _meta_seed(run_root / "run_2") == 2002


def test_base_seed_metamorphic_shift_preserves_per_run_offsets(tmp_path: Path) -> None:
    cfg_a = tmp_path / "cfg_a.yaml"
    cfg_b = tmp_path / "cfg_b.yaml"
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    _write_cfg(cfg_a, base_seed=100, runs=2, out_dir=out_a)
    _write_cfg(cfg_b, base_seed=900, runs=2, out_dir=out_b)

    batch_run(str(cfg_a))
    batch_run(str(cfg_b))
    root_a = _latest_run_root(out_a)
    root_b = _latest_run_root(out_b)

    a0, a1 = _meta_seed(root_a / "run_0"), _meta_seed(root_a / "run_1")
    b0, b1 = _meta_seed(root_b / "run_0"), _meta_seed(root_b / "run_1")
    assert a1 - a0 == 1
    assert b1 - b0 == 1
    assert b0 - a0 == 800
    assert b1 - a1 == 800


def test_base_seed_integration_parquet_uses_meta_run_seed(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    out_base = tmp_path / "out"
    _write_cfg(cfg_path, base_seed=314, runs=1, out_dir=out_base)

    batch_run(str(cfg_path))
    run_root = _latest_run_root(out_base)
    run_dir = run_root / "run_0"
    expected_seed = _meta_seed(run_dir)

    for name in ("steps.parquet", "area_steps.parquet", "agents.parquet", "votes.parquet"):
        df = pd.read_parquet(run_dir / name)
        if df.empty:
            continue
        seeds = set(df["run_seed"].astype(int).tolist())
        assert seeds == {expected_seed}, f"{name} has mismatched run_seed values: {seeds}"
