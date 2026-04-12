from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.logging.run_logger import RunLogger
from tests.factory import create_test_model


pytestmark = pytest.mark.phase1


def _model_one_area(**overrides):
    base = dict(
        seed=80,
        num_colors=3,
        num_agents=6,
        num_areas=1,
        height=8,
        width=8,
        av_area_height=8,
        av_area_width=8,
        area_size_variance=0.0,
        max_steps=2,
        # isolate altruism from rewards/costs
        election_cost_rate=0.0,
        reward_rate_personal=0.0,
        altruism_learning=False,
        altruism_static=0.25,
        altruism_init=0.9,  # should be ignored when learning is off
    )
    base.update(overrides)
    model, _ = create_test_model(**base)
    return model


def test_altruism_static_oracle_initializes_factor_when_learning_off() -> None:
    s = 0.3
    model = _model_one_area(altruism_learning=False, altruism_static=s, altruism_init=0.99)
    for a in model.voting_agents:
        if a is None:
            continue
        assert float(a.altruism_factor) == pytest.approx(s, abs=0.0)


def test_altruism_static_oracle_constant_over_steps_when_learning_off() -> None:
    s = 0.7
    model = _model_one_area(altruism_learning=False, altruism_static=s, max_steps=3)
    # Run a few steps; satisfaction signals may change but altruism should not.
    for _ in range(3):
        model.step()
    for a in model.voting_agents:
        if a is None:
            continue
        assert float(a.altruism_factor) == pytest.approx(s, abs=0.0)


def test_altruism_static_metamorphic_two_runs_change_in_mean_altruism() -> None:
    """Metamorphic: changing altruism_static changes the (constant) altruism_factor levels."""
    s1 = 0.1
    s2 = 0.9
    assert s2 > s1
    m1 = _model_one_area(seed=81, altruism_learning=False, altruism_static=s1)
    m2 = _model_one_area(seed=81, altruism_learning=False, altruism_static=s2)

    a1 = [float(a.altruism_factor) for a in m1.voting_agents if a is not None]
    a2 = [float(a.altruism_factor) for a in m2.voting_agents if a is not None]
    assert np.mean(a1) == pytest.approx(s1, abs=1e-12)
    assert np.mean(a2) == pytest.approx(s2, abs=1e-12)


def test_altruism_static_integration_logged_in_agents_and_steps(tmp_path: Path) -> None:
    """Integration: logged outputs must reflect altruism_static when learning is off."""
    s = 0.42
    model = _model_one_area(seed=82, altruism_learning=False, altruism_static=s, max_steps=1)

    logger = RunLogger(out_dir=tmp_path, run_seed=1, rule_idx=int(model.rule_idx), num_steps=1, store_grid=False)
    logger.attach_to_model(model)
    logger.write_static(model)
    logger.begin_step(1)
    model.step()
    logger.log_step(step=1, model=model, grid_snapshot=None)
    logger.end_step()
    logger.finalize()

    agents_df = pd.read_parquet(tmp_path / "agents.parquet")
    assert "altruism_factor" in agents_df.columns
    assert np.allclose(agents_df["altruism_factor"].to_numpy(dtype=float), s)

    steps_df = pd.read_parquet(tmp_path / "steps.parquet")
    assert "mean_altruism" in steps_df.columns
    assert float(steps_df.loc[0, "mean_altruism"]) == pytest.approx(s, abs=1e-6)


def test_altruism_static_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        _model_one_area(altruism_static=-0.01)
    with pytest.raises(ValueError):
        _model_one_area(altruism_static=1.01)
