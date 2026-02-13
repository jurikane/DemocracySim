from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from scripts.run_headless import batch_run
from src.config.loader import load_config


def _write_cfg(
    path: Path,
    *,
    out_dir: Path,
    runs: int,
    processes: int,
    base_seed: int,
    num_steps: int = 2,
    store_grid: bool = False,
    grid_interval: int = 1,
) -> None:
    cfg = load_config("toy.yaml").model_copy(deep=True)
    cfg.simulation.runs = int(runs)
    cfg.simulation.processes = int(processes)
    cfg.simulation.base_seed = int(base_seed)
    cfg.simulation.num_steps = int(num_steps)
    cfg.simulation.store_grid = bool(store_grid)
    cfg.simulation.grid_interval = int(grid_interval)
    payload = cfg.model_dump(mode="json")
    payload["output"] = {"directory": str(out_dir)}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _latest_run_root(base_out: Path) -> Path:
    roots = sorted([p for p in base_out.iterdir() if p.is_dir()])
    assert roots, f"No run root created under {base_out}"
    return roots[-1]


def _read_cfg_used(run_root: Path) -> dict:
    return yaml.safe_load((run_root / "config_used.yaml").read_text(encoding="utf-8"))


def _run_seed(run_dir: Path) -> int:
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
    return int(meta["run"]["run_seed"])


def test_runs_oracle_creates_exact_number_of_run_dirs(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    out_dir = tmp_path / "out"
    _write_cfg(cfg_path, out_dir=out_dir, runs=3, processes=1, base_seed=410)
    batch_run(str(cfg_path))

    run_root = _latest_run_root(out_dir)
    run_dirs = sorted([p.name for p in run_root.iterdir() if p.is_dir() and p.name.startswith("run_")])
    assert run_dirs == ["run_0", "run_1", "run_2"]
    assert _run_seed(run_root / "run_0") == 410
    assert _run_seed(run_root / "run_1") == 411
    assert _run_seed(run_root / "run_2") == 412


def test_runs_metamorphic_increasing_runs_adds_tail_run_with_expected_seed(tmp_path: Path) -> None:
    cfg_a = tmp_path / "cfg_a.yaml"
    cfg_b = tmp_path / "cfg_b.yaml"
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    _write_cfg(cfg_a, out_dir=out_a, runs=1, processes=1, base_seed=500)
    _write_cfg(cfg_b, out_dir=out_b, runs=2, processes=1, base_seed=500)

    batch_run(str(cfg_a))
    batch_run(str(cfg_b))
    root_a = _latest_run_root(out_a)
    root_b = _latest_run_root(out_b)

    assert (root_a / "run_0").exists()
    assert not (root_a / "run_1").exists()

    assert (root_b / "run_0").exists()
    assert (root_b / "run_1").exists()
    assert _run_seed(root_b / "run_0") == 500
    assert _run_seed(root_b / "run_1") == 501


def test_processes_integration_logged_and_behaviorally_neutral_for_sequential_runner(tmp_path: Path) -> None:
    """Current contract: batch_run is sequential; `processes` is config metadata only.

    If true parallel execution is implemented later, this test should be updated.
    """
    cfg_1 = tmp_path / "cfg_p1.yaml"
    cfg_4 = tmp_path / "cfg_p4.yaml"
    out_1 = tmp_path / "out_p1"
    out_4 = tmp_path / "out_p4"
    _write_cfg(cfg_1, out_dir=out_1, runs=1, processes=1, base_seed=700)
    _write_cfg(cfg_4, out_dir=out_4, runs=1, processes=4, base_seed=700)

    batch_run(str(cfg_1))
    batch_run(str(cfg_4))
    root_1 = _latest_run_root(out_1)
    root_4 = _latest_run_root(out_4)
    run_1 = root_1 / "run_0"
    run_4 = root_4 / "run_0"

    # Integration: config_used captures the knob value for provenance.
    used_1 = _read_cfg_used(root_1)
    used_4 = _read_cfg_used(root_4)
    assert int(used_1["simulation"]["processes"]) == 1
    assert int(used_4["simulation"]["processes"]) == 4

    # Current behavioral contract (sequential implementation): outputs stay identical.
    for name, sort_cols in [
        ("steps.parquet", ["step"]),
        ("area_steps.parquet", ["step", "area_id"]),
        ("agents.parquet", ["step", "agent_id"]),
        ("votes.parquet", ["step", "area_id", "agent_id"]),
    ]:
        a = pd.read_parquet(run_1 / name)
        b = pd.read_parquet(run_4 / name)
        cols = [c for c in sort_cols if c in a.columns and c in b.columns]
        if cols:
            a = a.sort_values(cols).reset_index(drop=True)
            b = b.sort_values(cols).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_h_knobs_interaction_runs_steps_and_grid_interval_in_batch_mode(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg_interact.yaml"
    out_dir = tmp_path / "out_interact"
    _write_cfg(
        cfg_path,
        out_dir=out_dir,
        runs=2,
        processes=1,
        base_seed=8800,
        num_steps=5,
        store_grid=True,
        grid_interval=2,
    )
    batch_run(str(cfg_path))
    run_root = _latest_run_root(out_dir)

    for run_id in (0, 1):
        run_dir = run_root / f"run_{run_id}"
        assert _run_seed(run_dir) == 8800 + run_id
        # Interaction expectation: with num_steps=5, interval=2 -> indices 0,1,3,5
        grids = sorted((run_dir / "grids").glob("grid_*.npy"))
        names = [p.name for p in grids]
        assert names == ["grid_0.npy", "grid_1.npy", "grid_3.npy", "grid_5.npy"]

        steps = pd.read_parquet(run_dir / "steps.parquet")
        assert set(steps["step"].tolist()) == {1, 2, 3, 4, 5}
