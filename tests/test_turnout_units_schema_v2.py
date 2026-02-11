from __future__ import annotations

from pathlib import Path

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

    # Contract: turnout == floor(participants/eligible * 100) (as implemented in Area.conduct_election()).
    eligible = area_steps["eligible_voters"].to_numpy(dtype=float)
    participants = area_steps["participants"].to_numpy(dtype=float)
    expected = np.where(eligible > 0, np.floor(participants / eligible * 100.0), 0.0)
    got = area_steps["turnout"].to_numpy(dtype=float)

    assert np.allclose(got, expected, atol=0.0), "turnout must be percent, derived from participants/eligible"


def test_steps_turnout_matches_mean_area_steps_turnout(v2_run_dir: Path) -> None:
    steps_path = v2_run_dir / "steps.parquet"
    area_steps_path = v2_run_dir / "area_steps.parquet"
    if not steps_path.exists() or not area_steps_path.exists():
        pytest.skip("schema v2 parquet artifacts not present")

    steps = pd.read_parquet(steps_path)
    area_steps = pd.read_parquet(area_steps_path)
    assert not steps.empty and not area_steps.empty

    mean_by_step = (
        area_steps.groupby("step", as_index=False)["turnout"].mean().rename(columns={"turnout": "expected_turnout"})
    )
    merged = steps.merge(mean_by_step, on="step", how="inner")
    assert not merged.empty

    # Both are percent values, so they should match closely.
    assert np.allclose(
        merged["turnout"].to_numpy(dtype=float),
        merged["expected_turnout"].to_numpy(dtype=float),
        atol=1e-6,
    )

