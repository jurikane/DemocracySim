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
    cfg_for_run.simulation.num_steps = 1
    cfg_for_run.simulation.runs = 1
    cfg_for_run.simulation.store_grid = False

    out_dir = tmp_path / "run"
    run_once(0, cfg_for_run, out_dir=out_dir)
    return out_dir


def test_static_json_includes_distance_function_mapping(v2_run_dir: Path) -> None:
    p = v2_run_dir / "static.json"
    assert p.exists()
    static = json.loads(p.read_text())

    df = static.get("distance_functions")
    assert isinstance(df, dict)

    names = df.get("names")
    impl = df.get("impl_names")
    assert isinstance(names, list) and names
    assert isinstance(impl, list) and impl
    assert len(names) == len(impl)
    assert all(isinstance(x, str) for x in names)
    assert all(isinstance(x, str) for x in impl)

    idx = df.get("selected_idx")
    assert isinstance(idx, int)
    assert 0 <= idx < len(names)
    assert df.get("selected_name") == names[idx]
    assert df.get("selected_impl_name") == impl[idx]

