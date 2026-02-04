from __future__ import annotations

import numpy as np

from tests.factory import create_test_model


def test_relative_delta_signal_is_computed_pre_asset_update_eps_1() -> None:
    """Contract: delta_rel is computed right before assets mutate.

    eps=1.0
    delta_rel = delta_abs / max(assets_pre, 1.0)

    This must be stored on the agent for learning + diagnostics.
    """

    model, _ = create_test_model(seed=123, num_agents=1, num_colors=3, num_areas=1)
    a = model.voting_agents[0]

    # Ensure deterministic known starting state
    a.assets = 10
    a.reset_reward_variables()
    a.add_common_reward(3)
    a.add_personal_reward(1)
    a.set_election_fee(2)

    delta_abs_expected = 3 + 1 - 2
    assets_pre = 10
    delta_rel_expected = delta_abs_expected / max(assets_pre, 1.0)

    a.reward_agent()

    assert np.isclose(a.election_delta_abs, delta_abs_expected)
    assert np.isclose(a.election_delta_rel, delta_rel_expected)
    assert np.isclose(a.assets, assets_pre + delta_abs_expected)
