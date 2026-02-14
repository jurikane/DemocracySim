from __future__ import annotations

import numpy as np

from tests.factory import create_test_model


def test_relative_delta_signal_is_computed_pre_asset_update() -> None:
    """Contract: delta_rel is computed from realized asset change.

    delta_rel = realized_delta_abs / assets_pre (if assets_pre > 0, else 0.0)

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
    delta_rel_expected = delta_abs_expected / assets_pre

    a.reward_agent()

    assert np.isclose(a.election_delta_abs, delta_abs_expected)
    assert np.isclose(a.election_delta_rel, delta_rel_expected)
    assert np.isclose(a.assets, assets_pre + delta_abs_expected)


def test_relative_delta_signal_scale_invariant_for_multiplicative_payoffs() -> None:
    """Scaling assets (and absolute fee/reward components) must not change delta_rel."""
    model, _ = create_test_model(seed=124, num_agents=1, num_colors=3, num_areas=1)
    a = model.voting_agents[0]

    k = 0.2  # target relative return
    for assets_pre in (0.4, 1.0, 10.0):
        a.assets = assets_pre
        a.reset_reward_variables()
        a.add_common_reward(assets_pre * 0.30)
        a.add_personal_reward(assets_pre * 0.10)
        a.set_election_fee(assets_pre * 0.20)
        # delta_abs = assets_pre * (0.30 + 0.10 - 0.20) = assets_pre * 0.20
        a.reward_agent()
        assert np.isclose(a.election_delta_rel, k, atol=1e-12)


def test_relative_delta_signal_has_no_artificial_kink_at_assets_1() -> None:
    """Near assets=1 there should be no threshold jump from normalization."""
    model, _ = create_test_model(seed=125, num_agents=1, num_colors=3, num_areas=1)
    a = model.voting_agents[0]

    k = -0.15
    vals = []
    for assets_pre in (0.99, 1.01):
        a.assets = assets_pre
        a.reset_reward_variables()
        # Choose components to realize delta_abs = assets_pre * k
        a.add_common_reward(assets_pre * 0.05)
        a.add_personal_reward(assets_pre * 0.00)
        a.set_election_fee(assets_pre * 0.20)
        a.reward_agent()
        vals.append(float(a.election_delta_rel))

    assert np.isclose(vals[0], k, atol=1e-12)
    assert np.isclose(vals[1], k, atol=1e-12)
    assert np.isclose(vals[0], vals[1], atol=1e-12)


def test_relative_delta_signal_zero_when_assets_pre_zero() -> None:
    model, _ = create_test_model(seed=126, num_agents=1, num_colors=3, num_areas=1)
    a = model.voting_agents[0]
    a.assets = 0.0
    a.reset_reward_variables()
    a.add_common_reward(1.0)
    a.add_personal_reward(0.0)
    a.set_election_fee(0.0)

    a.reward_agent()
    assert a.election_delta_rel == 0.0


def test_relative_delta_signal_uses_realized_delta_when_asset_floor_hits() -> None:
    """If raw delta would push assets below zero, delta_rel must use realized (clamped) delta."""
    model, _ = create_test_model(seed=127, num_agents=1, num_colors=3, num_areas=1)
    a = model.voting_agents[0]

    a.assets = 1.0
    a.reset_reward_variables()
    a.add_common_reward(0.0)
    a.add_personal_reward(0.0)
    a.set_election_fee(3.0)  # raw delta_abs = -3.0, realized delta_abs = -1.0 due floor

    a.reward_agent()

    assert np.isclose(a.assets, 0.0)
    assert np.isclose(a.election_delta_abs, -1.0)
    assert np.isclose(a.election_delta_rel, -1.0)
