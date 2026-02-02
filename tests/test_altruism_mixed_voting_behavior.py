from __future__ import annotations

import numpy as np

from tests.factory import create_test_model


def test_assumed_opt_dist_respects_altruism_extremes() -> None:
    """Contract: centralized distribution mixing respects altruism extremes.

    If altruism_factor==0 => target_dist == personal_opt_dist
    If altruism_factor==1 => target_dist == est_real_dist

    We test the centralized helper (single source of truth) rather than the legacy
    VoteAgent.compute_assumed_opt_dist implementation.
    """

    model, _cfg = create_test_model(seed=123, num_colors=3, num_agents=5)

    agent = model.voting_agents[0]
    assert agent is not None
    area = model.areas[0]
    assert area is not None

    # Match election pipeline semantics without depending on VoteAgent.update_known_cells.
    k = int(getattr(model, "known_cells", 0) or 0)
    agent.known_cells = list(area.cells[:k]) if len(area.cells) >= k else list(area.cells)

    est, _ = agent.estimate_real_distribution(area)
    personal = np.asarray(agent.personality, dtype=np.float32)

    from src.utils.ballots import mix_distributions

    mixed0 = mix_distributions(altruism_factor=0.0, est_real_dist=est, personal_opt_dist=personal)
    np.testing.assert_allclose(mixed0, personal, rtol=0, atol=1e-6)

    mixed1 = mix_distributions(altruism_factor=1.0, est_real_dist=est, personal_opt_dist=personal)
    np.testing.assert_allclose(mixed1, est, rtol=0, atol=1e-6)


def test_vote_returns_raw_oppose_scores_in_0_1() -> None:
    """VoteAgent.vote returns raw oppose-scores in [0,1] (no normalization)."""

    model, _cfg = create_test_model(seed=123, num_colors=3, num_agents=5)
    agent = model.voting_agents[0]
    assert agent is not None
    area = model.areas[0]
    assert area is not None

    # Match election pipeline semantics without depending on VoteAgent.update_known_cells.
    k = int(getattr(model, "known_cells", 0) or 0)
    agent.known_cells = list(area.cells[:k]) if len(area.cells) >= k else list(area.cells)

    oppose = np.asarray(agent.vote(area), dtype=np.float32)
    assert oppose.ndim == 1
    assert oppose.shape[0] == model.options.shape[0]
    assert np.all(np.isfinite(oppose))

    # Contract: distances are normalized to [0,1]
    assert float(np.min(oppose)) >= -1e-6
    assert float(np.max(oppose)) <= 1.0 + 1e-6

    # Not normalized: sum is not forced to 1
    s = float(np.sum(oppose))
    assert s >= 0.0
