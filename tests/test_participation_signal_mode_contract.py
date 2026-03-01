from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.factory import create_test_model
from src.agents.area import Area


def _fake_agent(
    *,
    group_idx: int,
    delta_rel: float,
    fee: float = 0.0,
    participating: bool = False,
    assets_pre: float = 100.0,
) -> SimpleNamespace:
    # Area signal builder reconstructs assets_pre via assets_post - delta_abs.
    delta_abs = delta_rel * assets_pre
    assets_post = assets_pre + delta_abs
    return SimpleNamespace(
        personality_group_idx=int(group_idx),
        _eligible_for_election=True,
        eligible_for_election=True,
        _participating=bool(participating),
        participating=bool(participating),
        _fee=float(fee),
        election_fee=float(fee),
        _delta_rel=float(delta_rel),
        election_delta_rel=float(delta_rel),
        _delta_abs=float(delta_abs),
        election_delta_abs=float(delta_abs),
        assets=float(assets_post),
    )


def _signal_model(**overrides) -> SimpleNamespace:
    cfg = dict(
        participation_signal_mode="group_centered_delta_rel_plus_fee",
        participation_signal_fee_weight=1.0,
        participation_signal_group_shrink_k=10.0,
        participation_signal_clip=10.0,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def test_raw_delta_rel_mode_matches_legacy_signal_exactly() -> None:
    agents = [
        _fake_agent(group_idx=0, delta_rel=0.10, participating=True, fee=1.0),
        _fake_agent(group_idx=0, delta_rel=-0.05, participating=False),
        _fake_agent(group_idx=1, delta_rel=0.02, participating=True, fee=0.5),
    ]
    model = _signal_model(participation_signal_mode="raw_delta_rel")

    signals = Area._compute_participation_learning_signals(agents, model)

    assert signals == pytest.approx([0.10, -0.05, 0.02], abs=1e-12)


def test_group_centered_signal_cancels_common_shift() -> None:
    agents_a = [
        _fake_agent(group_idx=0, delta_rel=0.10),
        _fake_agent(group_idx=0, delta_rel=0.10),
        _fake_agent(group_idx=1, delta_rel=-0.10),
        _fake_agent(group_idx=1, delta_rel=-0.10),
    ]
    agents_b = [
        _fake_agent(group_idx=0, delta_rel=0.30),
        _fake_agent(group_idx=0, delta_rel=0.30),
        _fake_agent(group_idx=1, delta_rel=0.10),
        _fake_agent(group_idx=1, delta_rel=0.10),
    ]
    model = _signal_model(participation_signal_group_shrink_k=0.0, participation_signal_clip=10.0)

    s_a = Area._compute_participation_learning_signals(agents_a, model)
    s_b = Area._compute_participation_learning_signals(agents_b, model)

    assert s_a == pytest.approx(s_b, abs=1e-12)


def test_fee_component_penalizes_only_participants() -> None:
    # Same group and same delta -> same centered component. Difference must be fee-only.
    agents = [
        _fake_agent(group_idx=0, delta_rel=0.20, participating=True, fee=1.0, assets_pre=100.0),
        _fake_agent(group_idx=0, delta_rel=0.20, participating=False, fee=0.0, assets_pre=100.0),
        _fake_agent(group_idx=1, delta_rel=0.00, participating=False),
        _fake_agent(group_idx=1, delta_rel=0.00, participating=False),
    ]
    model = _signal_model(participation_signal_group_shrink_k=0.0, participation_signal_fee_weight=1.0, participation_signal_clip=10.0)

    s = Area._compute_participation_learning_signals(agents, model)

    # fee_rel = 1 / 100 = 0.01; participant should have exactly lower signal by 0.01.
    assert (s[0] - s[1]) == pytest.approx(-0.01, abs=1e-12)


def test_group_shrinkage_reduces_small_group_signal_magnitude() -> None:
    agents = [
        _fake_agent(group_idx=0, delta_rel=0.30),
        _fake_agent(group_idx=1, delta_rel=-0.10),
        _fake_agent(group_idx=1, delta_rel=-0.10),
        _fake_agent(group_idx=1, delta_rel=-0.10),
        _fake_agent(group_idx=1, delta_rel=-0.10),
    ]
    no_shrink = _signal_model(participation_signal_group_shrink_k=0.0, participation_signal_clip=10.0)
    with_shrink = _signal_model(participation_signal_group_shrink_k=10.0, participation_signal_clip=10.0)

    s0 = Area._compute_participation_learning_signals(agents, no_shrink)
    s1 = Area._compute_participation_learning_signals(agents, with_shrink)

    # Agent 0 is the singleton group; its magnitude should be reduced by shrinkage.
    assert abs(s1[0]) < abs(s0[0])


def test_group_relative_party_mode_aligns_q_update_direction_with_group_relative_performance() -> None:
    # Group 0 outperforms group 1 by mean delta_rel.
    # Party mode signal should be group-relative (no action compensation in signal).
    agents = [
        _fake_agent(group_idx=0, delta_rel=0.20, participating=True),
        _fake_agent(group_idx=0, delta_rel=0.20, participating=False),
        _fake_agent(group_idx=1, delta_rel=-0.10, participating=True),
        _fake_agent(group_idx=1, delta_rel=-0.10, participating=False),
    ]
    model = _signal_model(
        participation_signal_mode="group_relative_delta_rel_party",
        participation_signal_group_shrink_k=0.0,
        participation_signal_clip=10.0,
    )

    s = Area._compute_participation_learning_signals(agents, model)

    # Expected centered group means: g0=+0.15, g1=-0.15
    assert s == pytest.approx([+0.15, +0.15, -0.15, -0.15], abs=1e-12)


def test_group_relative_party_mode_applies_participant_fee_penalty() -> None:
    # Same group and same delta_rel values; participant should get extra fee penalty
    # even in party mode.
    agents = [
        _fake_agent(group_idx=0, delta_rel=0.20, participating=True, fee=1.0, assets_pre=100.0),
        _fake_agent(group_idx=0, delta_rel=0.20, participating=False, fee=0.0, assets_pre=100.0),
        _fake_agent(group_idx=1, delta_rel=0.00, participating=False),
        _fake_agent(group_idx=1, delta_rel=0.00, participating=False),
    ]
    model = _signal_model(
        participation_signal_mode="group_relative_delta_rel_party",
        participation_signal_group_shrink_k=0.0,
        participation_signal_fee_weight=1.0,
        participation_signal_clip=10.0,
    )

    s = Area._compute_participation_learning_signals(agents, model)

    # centered_g0 = +0.1, centered_g1 = -0.1, fee_rel = 1/100 = 0.01
    # participant signal gets fee penalty.
    assert s == pytest.approx([+0.09, +0.10, -0.10, -0.10], abs=1e-12)


def test_party_mode_q_push_is_direct_signal_no_hidden_action_sign() -> None:
    agents = [
        _fake_agent(group_idx=0, delta_rel=0.20, participating=True, fee=1.0, assets_pre=100.0),
        _fake_agent(group_idx=0, delta_rel=0.20, participating=False),
        _fake_agent(group_idx=1, delta_rel=0.00, participating=True),
        _fake_agent(group_idx=1, delta_rel=0.00, participating=False),
    ]
    model = _signal_model(
        participation_signal_mode="group_relative_delta_rel_party",
        participation_signal_group_shrink_k=0.0,
        participation_signal_fee_weight=1.0,
        participation_signal_clip=10.0,
    )

    s, q = Area._compute_participation_learning_signals_and_q_pushes(agents, model)
    assert q == pytest.approx(s, abs=1e-12)


def test_participation_signal_mode_and_params_validate_in_model() -> None:
    model, _ = create_test_model(
        participation_signal_mode="group_centered_delta_rel_plus_fee",
        participation_signal_fee_weight=1.25,
        participation_signal_group_shrink_k=5.0,
        participation_signal_clip=0.4,
    )
    assert model.participation_signal_mode == "group_centered_delta_rel_plus_fee"
    assert float(model.participation_signal_fee_weight) == pytest.approx(1.25)
    assert float(model.participation_signal_group_shrink_k) == pytest.approx(5.0)
    assert float(model.participation_signal_clip) == pytest.approx(0.4)
    model_party, _ = create_test_model(participation_signal_mode="group_relative_delta_rel_party")
    assert model_party.participation_signal_mode == "group_relative_delta_rel_party"


def test_participation_signal_mode_param_validation_errors() -> None:
    with pytest.raises(ValueError):
        create_test_model(participation_signal_mode="unknown_mode")
    with pytest.raises(ValueError):
        create_test_model(participation_signal_fee_weight=-0.1)
    with pytest.raises(ValueError):
        create_test_model(participation_signal_group_shrink_k=-1.0)
    with pytest.raises(ValueError):
        create_test_model(participation_signal_clip=0.0)
