from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import pandas as pd
import pytest

from src.config.loader import load_config
from scripts.run_headless import run_once


pytestmark = pytest.mark.phase1


@pytest.fixture()
def v2_run_dir(tmp_path: Path) -> Path:
    """Create a tiny headless run so we can validate logged turnout units."""
    cfg = load_config("toy.yaml")
    cfg_for_run = cfg.model_copy(deep=True)
    cfg_for_run.simulation.num_steps = 4
    cfg_for_run.simulation.runs = 1
    cfg_for_run.simulation.grid_interval = 1
    cfg_for_run.simulation.store_grid = False  # not needed for turnout tests

    out_dir = tmp_path / "run"
    run_once(0, cfg_for_run, out_dir=out_dir)
    return out_dir


def test_area_steps_turnout_is_percent_and_matches_participants(v2_run_dir: Path) -> None:
    area_steps_path = v2_run_dir / "area_steps.parquet"
    if not area_steps_path.exists():
        pytest.skip("schema v2 parquet artifacts not present")

    area_steps = pd.read_parquet(area_steps_path)
    assert not area_steps.empty

    # Unit: percent (0..100).
    assert float(area_steps["turnout"].min()) >= 0.0
    assert float(area_steps["turnout"].max()) <= 100.0

    # Contract: turnout == floor(participants/area_num_agents * 100).
    static = json.loads((v2_run_dir / "static.json").read_text(encoding="utf-8"))
    area_info = ((static.get("personality_group_info") or {}).get("areas") or {})
    resident_by_area = {int(k): int(v.get("num_agents", 0)) for k, v in area_info.items()}
    participants = area_steps["participants"].to_numpy(dtype=float)
    resident = area_steps["area_id"].map(resident_by_area).fillna(0).to_numpy(dtype=float)
    expected = np.where(resident > 0, np.floor(participants / resident * 100.0), 0.0)
    got = area_steps["turnout"].to_numpy(dtype=float)

    assert np.allclose(got, expected, atol=0.0), "turnout must be percent, derived from participants/resident"


def test_steps_turnout_matches_population_weighted_area_turnout(v2_run_dir: Path) -> None:
    steps_path = v2_run_dir / "steps.parquet"
    area_steps_path = v2_run_dir / "area_steps.parquet"
    if not steps_path.exists() or not area_steps_path.exists():
        pytest.skip("schema v2 parquet artifacts not present")

    steps = pd.read_parquet(steps_path)
    area_steps = pd.read_parquet(area_steps_path)
    assert not steps.empty and not area_steps.empty

    static = json.loads((v2_run_dir / "static.json").read_text(encoding="utf-8"))
    area_info = ((static.get("personality_group_info") or {}).get("areas") or {})
    resident_total = float(sum(int(v.get("num_agents", 0)) for v in area_info.values()))
    weighted = area_steps.groupby("step", as_index=False)[["participants"]].sum()
    weighted["expected_turnout"] = np.where(
        resident_total > 0,
        100.0 * weighted["participants"] / resident_total,
        0.0,
    )
    merged = steps.merge(weighted[["step", "expected_turnout"]], on="step", how="inner")
    assert not merged.empty

    # Both are percent values, so they should match closely.
    assert np.allclose(
        merged["turnout"].to_numpy(dtype=float),
        merged["expected_turnout"].to_numpy(dtype=float),
        atol=1e-6,
    )
