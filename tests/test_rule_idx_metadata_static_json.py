from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config.loader import load_config
from scripts.run_headless import run_once


pytestmark = pytest.mark.phase1


@pytest.fixture()
def v2_run_dir(tmp_path: Path) -> Path:
    cfg = load_config("toy.yaml")
    cfg_for_run = cfg.model_copy(deep=True)
    cfg_for_run.simulation.num_steps = 2
    cfg_for_run.simulation.runs = 1
    cfg_for_run.simulation.store_grid = False

    out_dir = tmp_path / "run"
    run_once(0, cfg_for_run, out_dir=out_dir)
    return out_dir


def test_static_json_includes_rule_name_mapping(v2_run_dir: Path) -> None:
    p = v2_run_dir / "static.json"
    assert p.exists()
    static = json.loads(p.read_text())

    vr = static.get("voting_rules")
    assert isinstance(vr, dict)
    names = vr.get("names")
    assert isinstance(names, list)
    assert all(isinstance(x, str) for x in names)
    assert len(names) >= 2

    impl = vr.get("impl_names")
    assert isinstance(impl, list)
    assert all(isinstance(x, str) for x in impl)
    assert len(impl) == len(names)

    idx = vr.get("selected_idx")
    assert isinstance(idx, int)
    assert 0 <= idx < len(names)
    assert vr.get("selected_name") == names[idx]
    assert vr.get("selected_impl_name") == impl[idx]


def test_static_json_supports_random_rule_idx(tmp_path: Path) -> None:
    cfg = load_config("toy.yaml")
    cfg_for_run = cfg.model_copy(deep=True)
    cfg_for_run.model.rule_idx = 4
    cfg_for_run.simulation.num_steps = 2
    cfg_for_run.simulation.store_grid = False

    out_dir = tmp_path / "run_random"
    run_once(0, cfg_for_run, out_dir=out_dir)

    static = json.loads((out_dir / "static.json").read_text())
    vr = static.get("voting_rules") or {}
    names = vr.get("names") or []
    impl = vr.get("impl_names") or []
    idx = int(vr.get("selected_idx", -1))

    assert idx == 4
    assert len(names) >= 5
    assert len(impl) >= 5
    assert str(names[idx]).lower() == "random"
    assert str(impl[idx]) == "random_rule"
