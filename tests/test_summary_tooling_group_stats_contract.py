from __future__ import annotations

import pandas as pd

from src.analysis.summary_tooling import _build_area_group_series


def test_build_area_group_series_basic_counts_and_turnout() -> None:
    agents = pd.DataFrame(
        {
            "step": [1, 1, 1, 2, 2, 2],
            "agent_id": [0, 1, 2, 0, 1, 2],
            "personality_group_idx": [0, 1, 1, 0, 1, 1],
            "assets": [10.0, 4.0, 2.0, 9.0, 0.0, 3.0],
            "dissatisfaction_value": [0.2, 0.4, 0.7, 0.25, 0.5, 0.6],
        }
    )
    votes = pd.DataFrame(
        {
            "step": [1, 1, 1, 2],
            "area_id": [0, 0, 1, 0],
            "agent_id": [0, 1, 1, 0],
        }
    )
    area_agent_ids = {0: [0, 1], 1: [1, 2]}

    out = _build_area_group_series(agents=agents, votes=votes, area_agent_ids=area_agent_ids)
    assert not out.empty
    assert {"step", "area_id", "group_idx", "participants", "eligible", "residents", "turnout"}.issubset(out.columns)

    # area 0, step 2, group 1: one resident (agent 1), assets=0 => ineligible, non-participant.
    row = out[(out["step"] == 2) & (out["area_id"] == 0) & (out["group_idx"] == 1)].iloc[0]
    assert int(row["residents"]) == 1
    assert int(row["eligible"]) == 0
    assert int(row["participants"]) == 0
    assert float(row["turnout"]) == 0.0


def test_build_area_group_series_participant_share_sums_to_one_when_participants_present() -> None:
    agents = pd.DataFrame(
        {
            "step": [1, 1],
            "agent_id": [0, 1],
            "personality_group_idx": [0, 1],
            "assets": [10.0, 11.0],
            "dissatisfaction_value": [0.1, 0.2],
        }
    )
    votes = pd.DataFrame(
        {
            "step": [1, 1],
            "area_id": [0, 0],
            "agent_id": [0, 1],
        }
    )
    area_agent_ids = {0: [0, 1]}

    out = _build_area_group_series(agents=agents, votes=votes, area_agent_ids=area_agent_ids)
    block = out[(out["step"] == 1) & (out["area_id"] == 0)]
    s = float(block["participant_share"].sum())
    assert abs(s - 1.0) < 1e-6


def test_build_area_group_series_participant_abstainer_delta_and_fee_diagnostics() -> None:
    agents = pd.DataFrame(
        {
            "step": [1, 1, 1, 1],
            "agent_id": [0, 1, 2, 3],
            "personality_group_idx": [0, 0, 1, 1],
            "assets": [10.0, 10.0, 20.0, 20.0],
            "dissatisfaction_value": [0.2, 0.3, 0.4, 0.5],
            "participating": [True, False, True, False],
            "election_fee": [1.0, 0.0, 2.0, 0.0],
            "election_delta_rel": [0.10, -0.05, 0.20, -0.10],
        }
    )
    votes = pd.DataFrame(
        {
            "step": [1, 1],
            "area_id": [0, 0],
            "agent_id": [0, 2],
            "voted_altruistically": [False, True],
        }
    )
    area_agent_ids = {0: [0, 1, 2, 3]}

    out = _build_area_group_series(agents=agents, votes=votes, area_agent_ids=area_agent_ids)
    row_g0 = out[(out["step"] == 1) & (out["area_id"] == 0) & (out["group_idx"] == 0)].iloc[0]
    row_g1 = out[(out["step"] == 1) & (out["area_id"] == 0) & (out["group_idx"] == 1)].iloc[0]

    assert abs(float(row_g0["participants_mean_delta_rel"]) - 0.10) < 1e-6
    assert abs(float(row_g0["abstainers_mean_delta_rel"]) - (-0.05)) < 1e-6
    assert abs(float(row_g1["participants_mean_delta_rel"]) - 0.20) < 1e-6
    assert abs(float(row_g1["abstainers_mean_delta_rel"]) - (-0.10)) < 1e-6
    assert abs(float(row_g0["participants_mean_fee"]) - 1.0) < 1e-6
    assert abs(float(row_g1["participants_mean_fee"]) - 2.0) < 1e-6
    assert abs(float(row_g0["participants_mean_fee_over_assets"]) - 0.1) < 1e-6
    assert abs(float(row_g1["participants_mean_fee_over_assets"]) - 0.1) < 1e-6


def test_build_area_group_series_learning_signal_diagnostics() -> None:
    agents = pd.DataFrame(
        {
            "step": [1, 1, 1, 1],
            "agent_id": [0, 1, 2, 3],
            "personality_group_idx": [0, 0, 1, 1],
            "assets": [10.0, 10.0, 20.0, 20.0],
            "dissatisfaction_value": [0.2, 0.3, 0.4, 0.5],
            "participating": [True, False, True, False],
            "election_fee": [1.0, 0.0, 2.0, 0.0],
            "election_delta_rel": [0.10, -0.05, 0.20, -0.10],
            "participation_signal": [0.10, -0.02, 0.07, -0.03],
            "dissatisfaction_signal": [0.03, 0.01, -0.04, -0.02],
        }
    )
    votes = pd.DataFrame(
        {
            "step": [1, 1],
            "area_id": [0, 0],
            "agent_id": [0, 2],
            "voted_altruistically": [True, False],
        }
    )
    area_agent_ids = {0: [0, 1, 2, 3]}
    out = _build_area_group_series(agents=agents, votes=votes, area_agent_ids=area_agent_ids)
    row_g0 = out[(out["step"] == 1) & (out["area_id"] == 0) & (out["group_idx"] == 0)].iloc[0]
    row_g1 = out[(out["step"] == 1) & (out["area_id"] == 0) & (out["group_idx"] == 1)].iloc[0]

    assert abs(float(row_g0["participants_mean_participation_signal"]) - 0.10) < 1e-6
    assert abs(float(row_g0["abstainers_mean_participation_signal"]) - (-0.02)) < 1e-6
    assert abs(float(row_g1["participants_mean_participation_signal"]) - 0.07) < 1e-6
    assert abs(float(row_g1["abstainers_mean_participation_signal"]) - (-0.03)) < 1e-6
    assert abs(float(row_g0["group_mean_participation_q_update_proxy"]) - 0.06) < 1e-6
    assert abs(float(row_g1["group_mean_participation_q_update_proxy"]) - 0.05) < 1e-6
    assert abs(float(row_g0["altruistic_voters_mean_dissatisfaction_signal"]) - 0.03) < 1e-6
    assert pd.isna(row_g0["non_altruistic_voters_mean_dissatisfaction_signal"])
    assert pd.isna(row_g1["altruistic_voters_mean_dissatisfaction_signal"])
    assert abs(float(row_g1["non_altruistic_voters_mean_dissatisfaction_signal"]) - (-0.04)) < 1e-6
    assert abs(float(row_g0["altruistic_voters_mean_altruism_update_proxy"]) - 0.03) < 1e-6
    assert pd.isna(row_g0["non_altruistic_voters_mean_altruism_update_proxy"])
    assert pd.isna(row_g1["altruistic_voters_mean_altruism_update_proxy"])
    assert abs(float(row_g1["non_altruistic_voters_mean_altruism_update_proxy"]) - (-0.04)) < 1e-6
    assert pd.isna(row_g0["vote_mode_switch_share"])
    assert pd.isna(row_g1["vote_mode_switch_share"])


def test_build_area_group_series_vote_mode_switch_share() -> None:
    agents = pd.DataFrame(
        {
            "step": [1, 1, 1, 2, 2, 2],
            "agent_id": [0, 1, 2, 0, 1, 2],
            "personality_group_idx": [0, 0, 0, 0, 0, 0],
            "assets": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            "dissatisfaction_value": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            "participating": [True, True, False, True, True, True],
            "election_fee": [1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            "election_delta_rel": [0.01, 0.01, 0.0, 0.01, 0.01, 0.01],
            "participation_signal": [0.01, 0.01, 0.0, 0.01, 0.01, 0.01],
            "dissatisfaction_signal": [0.02, 0.02, 0.0, 0.02, 0.02, 0.02],
        }
    )
    votes = pd.DataFrame(
        {
            "step": [1, 1, 2, 2, 2],
            "area_id": [0, 0, 0, 0, 0],
            "agent_id": [0, 1, 0, 1, 2],
            "voted_altruistically": [True, False, False, False, True],
        }
    )
    area_agent_ids = {0: [0, 1, 2]}
    out = _build_area_group_series(agents=agents, votes=votes, area_agent_ids=area_agent_ids)
    row_t1 = out[(out["step"] == 1) & (out["area_id"] == 0) & (out["group_idx"] == 0)].iloc[0]
    row_t2 = out[(out["step"] == 2) & (out["area_id"] == 0) & (out["group_idx"] == 0)].iloc[0]
    assert abs(float(row_t1["vote_mode_switch_share"]) - 0.0) < 1e-6
    assert abs(float(row_t1["vote_mode_switch_from_altruistic_share"]) - 0.0) < 1e-6
    assert abs(float(row_t1["vote_mode_switch_from_non_altruistic_share"]) - 0.0) < 1e-6
    assert pd.isna(row_t1["participation_switch_to_abstain_share"])
    # Denominator is all participants at step 2 (3 participants: 0,1,2).
    # agent0 switched (True->False), agent1 stayed False, agent2 has no previous vote-mode.
    assert abs(float(row_t2["vote_mode_switch_share"]) - (1.0 / 3.0)) < 1e-6
    assert abs(float(row_t2["vote_mode_switch_from_altruistic_share"]) - (1.0 / 3.0)) < 1e-6
    assert abs(float(row_t2["vote_mode_switch_from_non_altruistic_share"]) - 0.0) < 1e-6
    # no agent switched from participating to abstaining at t2.
    assert abs(float(row_t2["participation_switch_to_abstain_share"]) - 0.0) < 1e-6
