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

