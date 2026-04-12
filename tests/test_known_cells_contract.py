from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from src.logging.run_logger import RunLogger
from tests.factory import create_test_model


def _force_all_participate(model) -> None:
    def _always(self, area):  # type: ignore[no-untyped-def]
        return True

    for a in model.voting_agents:
        a.ask_for_participation = types.MethodType(_always, a)


def _run_one_logged_step(model, out_dir):
    logger = RunLogger(
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
    return pd.read_parquet(out_dir / "votes.parquet")


def test_known_cells_validation_fail_loud():
    with pytest.raises(ValueError):
        create_test_model(known_cells=-1)
    with pytest.raises(ValueError):
        create_test_model(known_cells=1.5)
    with pytest.raises(ValueError):
        create_test_model(known_cells=True)


def test_known_cells_oracle_zero_yields_empty_knowledge_after_update():
    model, _ = create_test_model(
        seed=1401,
        num_agents=1,
        known_cells=0,
        num_areas=1,
        height=10,
        width=10,
        av_area_height=10,
        av_area_width=10,
        area_size_variance=0.0,
    )
    agent = model.voting_agents[0]
    area = model.areas[0]
    agent.update_known_cells(area)
    assert len(agent.known_cells) == 0
    dist, conf = agent.estimate_real_distribution(area)
    np.testing.assert_allclose(float(conf), 0.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.sum(dist), 1.0, rtol=0.0, atol=1e-12)


def test_known_cells_metamorphic_increases_confidence():
    base_kwargs = dict(
        seed=1402,
        num_agents=1,
        num_areas=1,
        height=10,
        width=10,
        av_area_height=10,
        av_area_width=10,
        area_size_variance=0.0,
    )
    m1, _ = create_test_model(**base_kwargs, known_cells=2)
    m2, _ = create_test_model(**base_kwargs, known_cells=8)
    a1, area1 = m1.voting_agents[0], m1.areas[0]
    a2, area2 = m2.voting_agents[0], m2.areas[0]
    a1.update_known_cells(area1)
    a2.update_known_cells(area2)
    _, c1 = a1.estimate_real_distribution(area1)
    _, c2 = a2.estimate_real_distribution(area2)
    assert c2 > c1
    np.testing.assert_allclose(c1, 2.0 / area1.num_cells, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(c2, 8.0 / area2.num_cells, rtol=0.0, atol=1e-12)


def test_known_cells_integration_logged_vote_confidence_scales(tmp_path):
    base_kwargs = dict(
        seed=1403,
        num_agents=4,
        num_areas=1,
        num_colors=3,
        height=10,
        width=10,
        av_area_height=10,
        av_area_width=10,
        area_size_variance=0.0,
        max_steps=1,
        election_cost_rate=0.0,
        reward_rate_personal=0.0,
    )
    m0, _ = create_test_model(**base_kwargs, known_cells=0)
    m5, _ = create_test_model(**base_kwargs, known_cells=5)
    _force_all_participate(m0)
    _force_all_participate(m5)

    v0 = _run_one_logged_step(m0, tmp_path / "k0")
    v5 = _run_one_logged_step(m5, tmp_path / "k5")
    assert len(v0) > 0 and len(v5) > 0

    c0 = float(v0["confidence"].mean())
    c5 = float(v5["confidence"].mean())
    np.testing.assert_allclose(c0, 0.0, rtol=0.0, atol=1e-8)
    assert c5 > c0

