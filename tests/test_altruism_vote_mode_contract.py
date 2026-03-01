from __future__ import annotations

import types

import numpy as np
import pandas as pd

from src.logging.run_logger import RunLoggerV2
from tests.factory import create_test_model


def _single_agent_model(seed: int = 9011):
    model, _ = create_test_model(
        seed=seed,
        num_agents=1,
        num_areas=1,
        num_colors=4,
        num_personality_groups=4,
        height=8,
        width=8,
        av_area_height=8,
        av_area_width=8,
        area_size_variance=0.0,
        known_cells=8,
        max_steps=1,
    )
    area = model.areas[0]
    agent = model.voting_agents[0]
    return model, area, agent


def test_altruism_vote_mode_extremes_are_deterministic() -> None:
    _, area, agent = _single_agent_model(seed=9012)

    # altruism_factor=0 -> always self-regarding and must return cached scores.
    agent.altruism_factor = 0.0
    agent.update_known_cells(area)
    s0 = np.asarray(agent.vote(area), dtype=np.float32)
    assert agent.voted_altruistically is False
    assert np.allclose(s0, np.asarray(agent.self_regarding_oppose_scores, dtype=np.float32), atol=1e-12)

    # altruism_factor=1 -> always altruistic.
    reverse_order = np.asarray(agent.personality_group, dtype=np.int64)[::-1]
    dist = np.zeros(agent.model.num_colors, dtype=np.float32)
    for rank, color in enumerate(reverse_order):
        dist[int(color)] = np.float32(agent.model.num_colors - rank)
    dist /= np.sum(dist)

    def _fake_est(self, _area):  # type: ignore[no-untyped-def]
        self.est_real_dist = np.asarray(dist, dtype=np.float32)
        self.confidence = 1.0
        return self.est_real_dist, self.confidence

    agent.estimate_real_distribution = types.MethodType(_fake_est, agent)
    agent.altruism_factor = 1.0
    s1 = np.asarray(agent.vote(area), dtype=np.float32)
    assert agent.voted_altruistically is True
    assert not np.allclose(s1, np.asarray(agent.self_regarding_oppose_scores, dtype=np.float32), atol=1e-9)


def test_altruism_vote_mode_sequence_is_seed_reproducible() -> None:
    m1, a1, g1 = _single_agent_model(seed=9013)
    m2, a2, g2 = _single_agent_model(seed=9013)
    g1.altruism_factor = 0.37
    g2.altruism_factor = 0.37

    seq1: list[bool] = []
    seq2: list[bool] = []
    for _ in range(80):
        g1.update_known_cells(a1)
        g2.update_known_cells(a2)
        _ = g1.vote(a1)
        _ = g2.vote(a2)
        seq1.append(bool(g1.voted_altruistically))
        seq2.append(bool(g2.voted_altruistically))

    assert seq1 == seq2
    altruistic_share = float(np.mean([1.0 if x else 0.0 for x in seq1]))
    assert 0.15 <= altruistic_share <= 0.60


def test_votes_parquet_logs_voted_altruistically(tmp_path) -> None:
    model, _ = create_test_model(
        seed=9014,
        num_agents=6,
        num_areas=1,
        num_colors=4,
        num_personality_groups=4,
        height=10,
        width=10,
        av_area_height=10,
        av_area_width=10,
        area_size_variance=0.0,
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

    votes = pd.read_parquet(tmp_path / "votes.parquet")
    assert "voted_altruistically" in votes.columns
    if not votes.empty:
        observed = set(votes["voted_altruistically"].dropna().astype(bool).unique().tolist())
        assert observed.issubset({True, False})
