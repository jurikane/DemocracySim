from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from src.analysis.doe_runner import DOERunTask
from scripts.run_doe import execute_run_plan
from src.config.loader import load_config


def test_execute_run_plan_continue_on_error(tmp_path: Path) -> None:
    cfg = load_config("test_small.yaml")
    tasks = [
        DOERunTask(design_id=0, seed=101, rule_idx=1, out_dir=tmp_path / "a", params={}),
        DOERunTask(design_id=0, seed=202, rule_idx=1, out_dir=tmp_path / "b", params={}),
    ]
    calls: list[int] = []

    def _fake_run_once(*, run_id: int, cfg, out_dir: Path) -> None:
        calls.append(int(cfg.simulation.base_seed))
        if int(cfg.simulation.base_seed) == 101:
            raise RuntimeError("boom")

    summary = execute_run_plan(
        cfg=cfg,
        plan=tasks,
        run_once_fn=_fake_run_once,
        continue_on_error=True,
    )
    assert summary["succeeded"] == 1
    assert summary["failed"] == 1
    assert calls == [101, 202]


def test_execute_run_plan_raises_without_continue(tmp_path: Path) -> None:
    cfg = load_config("test_small.yaml")
    tasks = [
        DOERunTask(design_id=0, seed=101, rule_idx=1, out_dir=tmp_path / "a", params={}),
    ]

    def _fake_run_once(*, run_id: int, cfg, out_dir: Path) -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        execute_run_plan(
            cfg=cfg,
            plan=tasks,
            run_once_fn=_fake_run_once,
            continue_on_error=False,
        )


def test_run_doe_dry_run_uses_default_doe_config(tmp_path: Path) -> None:
    out_root = tmp_path / "doe_dry"
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_doe",
        "--points",
        "1",
        "--seed-mode",
        "fixed",
        "--seeds",
        "101",
        "--no-robustness",
        "--dry-run",
        "--out-root",
        str(out_root),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert (out_root / "doe_spec.json").exists()
    assert (out_root / "doe_seed_selection.json").exists()
    assert (out_root / "doe_run_manifest.csv").exists()
