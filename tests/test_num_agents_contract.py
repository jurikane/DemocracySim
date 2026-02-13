from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.logging.run_logger import RunLoggerV2
from tests.factory import create_test_model


def test_num_agents_validation_fail_loud():
    with pytest.raises(ValueError):
        create_test_model(num_agents=0)
    with pytest.raises(ValueError):
        create_test_model(num_agents=-1)
    with pytest.raises(ValueError):
        create_test_model(num_agents=1.5)
    with pytest.raises(ValueError):
        create_test_model(num_agents=True)


def test_num_agents_oracle_initialization_count_and_assets():
    n = 7
    model, _ = create_test_model(num_agents=n, initial_agent_assets=100.0, seed=1201)
    assert model.num_agents == n
    assert len(model.voting_agents) == n
    assert model.initial_agent_assets == 100.0

    assets = np.array([float(a.assets) for a in model.voting_agents], dtype=np.float64)
    assert len(assets) == n
    np.testing.assert_allclose(assets, np.full(n, 100.0), rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(float(np.sum(assets)), 100.0 * n, rtol=0.0, atol=1e-8)


def test_num_agents_metamorphic_total_initial_assets_scale_linearly():
    n1, n2 = 5, 11
    m1, _ = create_test_model(num_agents=n1, initial_agent_assets=100.0, seed=1202)
    m2, _ = create_test_model(num_agents=n2, initial_agent_assets=100.0, seed=1202)
    total1 = float(np.sum([float(a.assets) for a in m1.voting_agents]))
    total2 = float(np.sum([float(a.assets) for a in m2.voting_agents]))
    assert total2 / total1 == n2 / n1


def test_num_agents_integration_logging_agents_and_area_rows(tmp_path):
    n = 9
    model, _ = create_test_model(
        num_agents=n,
        num_areas=1,
        height=10,
        width=10,
        av_area_height=10,
        av_area_width=10,
        area_size_variance=0.0,
        num_colors=3,
        seed=1203,
        max_steps=1,
    )

    logger = RunLoggerV2(
        out_dir=tmp_path,
        run_seed=1,
        rule_idx=int(model.rule_idx),
        num_steps=1,
        store_grid=False,
    )
    logger.attach_to_model(model)
    logger.begin_step(1)
    model.step()
    logger.log_step(step=1, model=model, grid_snapshot=None)
    logger.end_step()
    logger.finalize()

    agents_df = pd.read_parquet(tmp_path / "agents.parquet")
    area_steps_df = pd.read_parquet(tmp_path / "area_steps.parquet")
    steps_df = pd.read_parquet(tmp_path / "steps.parquet")

    # One agent row per agent for step 1.
    assert len(agents_df) == n
    assert int(agents_df["agent_id"].nunique()) == n

    # Full-coverage single area run: eligible_voters equals num_agents.
    assert len(area_steps_df) == 1
    assert int(area_steps_df.iloc[0]["eligible_voters"]) == n

    # Global turnout remains in schema percent units.
    assert len(steps_df) == 1
    turnout = float(steps_df.iloc[0]["turnout"])
    assert 0.0 <= turnout <= 100.0
