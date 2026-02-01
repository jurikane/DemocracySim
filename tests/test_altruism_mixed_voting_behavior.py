from __future__ import annotations

import numpy as np

from tests.factory import create_test_model


def test_assumed_opt_dist_respects_altruism_extremes() -> None:
    """Batch 3: behavior-only contract.

    If altruism_factor==0 => assumed_opt ~= personal_opt_dist
    If altruism_factor==1 => assumed_opt ~= est_real_dist

    We inject a deterministic altruism policy for the test.
    """

    model, _cfg = create_test_model(seed=123, num_colors=3, num_agents=5)

    # pick one agent and one area
    agent = model.voting_agents[0]
    area = model.areas[0]

    # ensure known cells get updated so estimate_real_distribution isn't using Nones
    agent.update_known_cells(area)
    est, _ = agent.estimate_real_distribution(area)
    personal = np.asarray(agent.personal_opt_dist, dtype=np.float32)

    class _AlwaysZero:
        def decide_participation(self, agent, area) -> bool:
            return True

        def decide_altruism_factor(self, agent, area) -> float:
            return 0.0

        def rank_options(self, agent, area, options):
            # Not used by compute_assumed_opt_dist
            return np.ones(options.shape[0], dtype=np.float32) / float(options.shape[0])

    class _AlwaysOne(_AlwaysZero):
        def decide_altruism_factor(self, agent, area) -> float:
            return 1.0

    agent.policy = _AlwaysZero()
    assumed0 = np.asarray(agent.compute_assumed_opt_dist(area), dtype=np.float32)
    np.testing.assert_allclose(assumed0, personal, rtol=0, atol=1e-6)

    agent.policy = _AlwaysOne()
    assumed1 = np.asarray(agent.compute_assumed_opt_dist(area), dtype=np.float32)
    np.testing.assert_allclose(assumed1, est, rtol=0, atol=1e-6)


def test_vote_returns_normalized_oppose_scores() -> None:
    model, _cfg = create_test_model(seed=123, num_colors=3, num_agents=5)
    agent = model.voting_agents[0]
    area = model.areas[0]

    # Force participation + fixed altruism factor to keep predictable.
    class _Fixed:
        def decide_participation(self, agent, area) -> bool:
            return True

        def decide_altruism_factor(self, agent, area) -> float:
            return 0.5

        def rank_options(self, agent, area, options):
            r = np.ones(options.shape[0], dtype=np.float32)
            r /= float(r.sum())
            return r

    agent.policy = _Fixed()

    # ensure known cells set up for estimate_real_distribution
    agent.update_known_cells(area)

    oppose = np.asarray(agent.vote(area), dtype=np.float32)
    assert oppose.ndim == 1
    assert oppose.shape[0] == model.options.shape[0]
    assert np.all(np.isfinite(oppose))
    assert np.all(oppose >= 0)

    s = float(oppose.sum())
    # If all distances are zero (edge case), vote() may return all zeros.
    assert abs(s - 1.0) < 1e-6 or abs(s - 0.0) < 1e-6
