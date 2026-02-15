from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.logging.run_logger import RunLoggerV2
from tests.factory import create_test_model


pytestmark = pytest.mark.phase1


def _model_one_area(**overrides):
    base = dict(
        seed=90,
        num_colors=3,
        num_agents=2,
        num_areas=1,
        height=8,
        width=8,
        av_area_height=8,
        av_area_width=8,
        area_size_variance=0.0,
        max_steps=1,
        # ensure altruism is learnable and unclipped
        altruism_learning=True,
        altruism_init=0.5,
        altruism_clip_min=0.0,
        altruism_clip_max=1.0,
        # isolate other dynamics
        known_cells=0,
        mu=0.0,
        election_cost_rate=0.0,
        reward_rate_common=0.0,
        reward_rate_personal=0.0,
    )
    base.update(overrides)
    model, _ = create_test_model(**base)
    return model


def test_altruism_alpha_oracle_exact_update_without_clipping() -> None:
    alpha = 0.2
    sig = 0.5
    model = _model_one_area(altruism_alpha=alpha)
    a = model.voting_agents[0]
    assert a is not None

    a._participating = True
    a.altruism_factor = 0.6
    a.apply_altruism_update(dissatisfaction_signal=sig)
    assert float(a.altruism_factor) == pytest.approx(0.6 + alpha * sig, abs=1e-12)


def test_altruism_alpha_metamorphic_ratio_two_runs() -> None:
    """Metamorphic: scaling altruism_alpha scales delta-a proportionally (away from clipping)."""
    a1 = 0.1
    a2 = 0.4
    assert a2 > a1
    ratio = a2 / a1
    sig = 0.25

    m1 = _model_one_area(seed=91, altruism_alpha=a1, altruism_clip_min=0.0, altruism_clip_max=10.0, altruism_init=1.0)
    m2 = _model_one_area(seed=91, altruism_alpha=a2, altruism_clip_min=0.0, altruism_clip_max=10.0, altruism_init=1.0)
    x1 = m1.voting_agents[0]
    x2 = m2.voting_agents[0]
    assert x1 is not None and x2 is not None

    x1._participating = True
    x2._participating = True
    x1.altruism_factor = 0.5
    x2.altruism_factor = 0.5

    x1.apply_altruism_update(dissatisfaction_signal=sig)
    x2.apply_altruism_update(dissatisfaction_signal=sig)

    da1 = float(x1.altruism_factor) - 0.5
    da2 = float(x2.altruism_factor) - 0.5
    assert (da2 / da1) == pytest.approx(ratio, rel=1e-12, abs=1e-12)


def test_altruism_alpha_zero_means_no_update() -> None:
    model = _model_one_area(seed=92, altruism_alpha=0.0)
    a = model.voting_agents[0]
    assert a is not None
    a._participating = True
    a.altruism_factor = 0.6
    a.apply_altruism_update(dissatisfaction_signal=999.0)
    assert float(a.altruism_factor) == pytest.approx(0.6, abs=0.0)


def test_altruism_alpha_integration_logged_mean_altruism_changes_after_step(tmp_path: Path) -> None:
    """Integration: with learning on and a forced positive dissatisfaction_signal on step 2,
    schema v2 logging should show increased altruism.

    We patch compute_dissatisfaction_value so:
    - step 1: sv=0.0 => baseline initializes, signal=0.0 (no altruism update)
    - step 2: sv=1.0 and satisfaction_baseline_alpha=0 => baseline stays 0, signal=+1.0
      => altruism_factor increases by altruism_alpha for participants.
    """
    alpha = 0.5
    init_a = 0.5
    model = _model_one_area(
        seed=93,
        altruism_alpha=alpha,
        altruism_init=init_a,
        satisfaction_baseline_alpha=0.0,
        max_steps=2,
    )
    area = model.areas[0]

    # Ensure the election runs (at least one participant) and avoid voting randomness.
    class _AlwaysParticipate:
        def decide_participation(self, agent, area) -> bool:  # type: ignore[no-untyped-def]
            return True

    class _ZeroBallot:
        def score_options(self, agent, area, options) -> np.ndarray:  # type: ignore[no-untyped-def]
            return np.zeros(int(options.shape[0]), dtype=np.float32)

    for a in model.voting_agents:
        if a is None:
            continue
        a.participation_strategy = _AlwaysParticipate()
        a.voting_strategy = _ZeroBallot()

    # Patch satisfaction value as a function of scheduler step counter.
    # Area.step computes satisfaction before conduct_election increments any counters.
    def _sv(_self, *, area, model) -> float:  # type: ignore[no-untyped-def]
        # scheduler.steps is 1 on the first recorded step in this model.
        return 0.0 if int(model.scheduler.steps) <= 1 else 1.0

    for a in model.voting_agents:
        if a is None:
            continue
        a.compute_dissatisfaction_value = _sv.__get__(a, type(a))  # bind method

    logger = RunLoggerV2(out_dir=tmp_path, run_seed=1, rule_idx=int(model.rule_idx), num_steps=2, store_grid=False)
    logger.attach_to_model(model)
    logger.write_static(model)

    # Step 1
    logger.begin_step(1)
    model.step()
    logger.log_step(step=1, model=model, grid_snapshot=None)
    logger.end_step()

    # Step 2 (expected altruism increase)
    logger.begin_step(2)
    model.step()
    logger.log_step(step=2, model=model, grid_snapshot=None)
    logger.end_step()
    logger.finalize()

    agents_df = pd.read_parquet(tmp_path / "agents.parquet")
    a2 = agents_df[agents_df["step"] == 2]["altruism_factor"].to_numpy(dtype=float)
    assert a2.size > 0
    assert np.allclose(a2, init_a + alpha * 1.0, atol=1e-6)

    steps_df = pd.read_parquet(tmp_path / "steps.parquet")
    row2 = steps_df[steps_df["step"] == 2].iloc[0]
    assert float(row2["mean_altruism"]) == pytest.approx(init_a + alpha * 1.0, abs=1e-6)


def test_altruism_alpha_negative_raises() -> None:
    with pytest.raises(ValueError, match="Learning rates alpha must be finite and >= 0"):
        _model_one_area(altruism_alpha=-0.01)
