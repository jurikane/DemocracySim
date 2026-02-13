from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.logging.run_logger import RunLoggerV2
from tests.factory import create_test_model


def _initial_assets(model) -> np.ndarray:
    return np.array([float(a.assets) for a in model.voting_agents], dtype=np.float64)


def _run_one_logged_step(model, out_dir):
    logger = RunLoggerV2(
        out_dir=out_dir,
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
    return pd.read_parquet(out_dir / "agents.parquet")


def test_initial_agent_assets_validation_fail_loud():
    with pytest.raises(ValueError):
        create_test_model(initial_agent_assets=-1)
    with pytest.raises(ValueError):
        create_test_model(initial_agent_assets=float("nan"))
    with pytest.raises(ValueError):
        create_test_model(initial_agent_assets=float("inf"))
    with pytest.raises(ValueError):
        create_test_model(initial_agent_assets=True)


def test_initial_agent_assets_oracle_initial_distribution_exact():
    n = 8
    per_agent = 200.0
    model, _ = create_test_model(num_agents=n, initial_agent_assets=per_agent, seed=1301)
    assert float(model.initial_agent_assets) == per_agent
    assets = _initial_assets(model)
    np.testing.assert_allclose(assets, np.full(n, per_agent), rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(float(np.sum(assets)), per_agent * n, rtol=0.0, atol=1e-8)


def test_initial_agent_assets_metamorphic_doubles_per_agent_assets():
    n = 8
    m1, _ = create_test_model(num_agents=n, initial_agent_assets=100.0, seed=1302)
    m2, _ = create_test_model(num_agents=n, initial_agent_assets=200.0, seed=1302)
    a1 = _initial_assets(m1)
    a2 = _initial_assets(m2)
    np.testing.assert_allclose(a2, 2.0 * a1, rtol=0.0, atol=1e-8)


def test_initial_agent_assets_integration_logged_assets_scale_with_per_agent_value(tmp_path):
    # Disable asset-changing mechanics so logged assets remain pure initialization outcomes.
    base_kwargs = dict(
        seed=1303,
        num_agents=10,
        num_areas=1,
        height=10,
        width=10,
        av_area_height=10,
        av_area_width=10,
        area_size_variance=0.0,
        max_steps=1,
        election_cost_rate=0.0,
        reward_rate_common=0.0,
        reward_rate_personal=0.0,
    )
    m1, _ = create_test_model(**base_kwargs, initial_agent_assets=100.0)
    m2, _ = create_test_model(**base_kwargs, initial_agent_assets=200.0)

    df1 = _run_one_logged_step(m1, tmp_path / "r1")
    df2 = _run_one_logged_step(m2, tmp_path / "r2")

    assert len(df1) == len(df2) == 10
    mean1 = float(df1["assets"].mean())
    mean2 = float(df2["assets"].mean())
    np.testing.assert_allclose(mean2, 2.0 * mean1, rtol=0.0, atol=1e-6)

