from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.factory import create_test_model
from src.logging.run_logger import RunLogger


pytestmark = pytest.mark.phase1


class _AlwaysParticipate:
    def decide_participation(self, agent, area) -> bool:  # type: ignore[no-untyped-def]
        return True


class _ZeroBallot:
    def score_options(self, agent, area, options) -> np.ndarray:  # type: ignore[no-untyped-def]
        # Valid ScoreVector in [0,1].
        return np.zeros(int(options.shape[0]), dtype=np.float32)


def _model_all_agents_in_one_area(**overrides):
    # Geometry: 1 area covers entire grid so every agent is in that area's agent list.
    base = dict(
        num_colors=3,
        num_agents=10,
        num_areas=1,
        height=8,
        width=8,
        av_area_height=8,
        av_area_width=8,
        area_size_variance=0.0,
        max_steps=2,
    )
    base.update(overrides)
    return create_test_model(**base)[0]


def test_election_cost_rate_charges_only_participants_and_updates_fee_pool() -> None:
    rate = 0.1
    model = _model_all_agents_in_one_area(
        seed=123,
        election_cost_rate=rate,
        reward_rate_personal=0.0,
    )

    # Force everyone to participate and cast a valid ballot.
    for a in model.voting_agents:
        if a is None:
            continue
        a.participation_strategy = _AlwaysParticipate()
        a.voting_strategy = _ZeroBallot()

    area = model.areas[0]
    assert area is not None
    assert area.agents, "Expected the area to contain agents"

    assets0 = {int(a.unique_id): float(a.assets) for a in area.agents}

    model.step()

    # All participate => turnout 100, participants == eligible.
    assert int(area.voter_turnout) == 100
    assert int(area.num_agents_participated_last) == int(area.num_agents)

    expected_fees = [assets0[int(a.unique_id)] * rate for a in area.agents]
    expected_fee_pool = float(sum(expected_fees))
    assert float(area._election_fee_pool) == pytest.approx(expected_fee_pool, abs=1e-9)

    # With reward rates 0, delta_abs == -fee, so assets are scaled by (1-rate).
    for a in area.agents:
        a0 = assets0[int(a.unique_id)]
        assert float(a.assets) == pytest.approx(a0 * (1.0 - rate), abs=1e-9)


def test_election_cost_rate_zero_means_no_fee_pool_and_no_asset_change_from_fee() -> None:
    model = _model_all_agents_in_one_area(
        seed=321,
        election_cost_rate=0.0,
        reward_rate_personal=0.0,
    )
    for a in model.voting_agents:
        if a is None:
            continue
        a.participation_strategy = _AlwaysParticipate()
        a.voting_strategy = _ZeroBallot()

    area = model.areas[0]
    assets0 = [float(a.assets) for a in area.agents]
    model.step()
    assert float(area._election_fee_pool) == pytest.approx(0.0, abs=0.0)
    assert [float(a.assets) for a in area.agents] == pytest.approx(assets0, abs=1e-12)


def test_election_cost_rate_allows_fractional_fees_no_minimum_fee_regression() -> None:
    """Regression guard: no hidden `min fee = 1` clamp.

    Fee is purely proportional: fee = election_cost_rate * assets.
    """
    rate = 0.1
    model = _model_all_agents_in_one_area(
        seed=999,
        num_agents=1,
        election_cost_rate=rate,
        reward_rate_personal=0.0,
        max_steps=1,
    )
    agent = next(a for a in model.voting_agents if a is not None)
    agent.assets = 0.5  # small assets => fee must be < 1.0 if not clamped
    agent.participation_strategy = _AlwaysParticipate()
    agent.voting_strategy = _ZeroBallot()

    area = model.areas[0]
    model.step()

    assert float(getattr(agent, "_fee")) == pytest.approx(0.05, abs=1e-12)
    assert float(area._election_fee_pool) == pytest.approx(0.05, abs=1e-12)
    assert float(agent.assets) == pytest.approx(0.45, abs=1e-12)


def test_election_cost_rate_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="Rate value must be in \[0,1\]."):
        _model_all_agents_in_one_area(election_cost_rate=-0.01)
    with pytest.raises(ValueError, match="Rate value must be in \[0,1\]."):
        _model_all_agents_in_one_area(election_cost_rate=1.01)


def test_area_steps_logs_fee_pool(tmp_path: Path) -> None:
    rate = 0.2
    model = _model_all_agents_in_one_area(
        seed=777,
        election_cost_rate=rate,
        reward_rate_personal=0.0,
        max_steps=1,
    )
    for a in model.voting_agents:
        if a is None:
            continue
        a.participation_strategy = _AlwaysParticipate()
        a.voting_strategy = _ZeroBallot()

    area = model.areas[0]
    assets0 = {int(a.unique_id): float(a.assets) for a in area.agents}
    expected_fee_pool = float(sum(v * rate for v in assets0.values()))

    logger = RunLogger(out_dir=tmp_path, run_seed=1, rule_idx=int(model.rule_idx), num_steps=1, store_grid=False)
    logger.attach_to_model(model)
    logger.begin_step(1)
    model.step()
    logger.log_step(step=1, model=model, grid_snapshot=None)
    logger.end_step()

    rows = [r for r in logger._area_steps_rows if int(r["area_id"]) == int(area.unique_id)]
    assert rows
    row = rows[0]
    assert float(row["fee_pool"]) == pytest.approx(expected_fee_pool, abs=1e-6)
