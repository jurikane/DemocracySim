from __future__ import annotations

import json
import types

import numpy as np
import pandas as pd

from src.logging.run_logger import RunLoggerV2
from tests.factory import create_test_model


def _force_all_participate(model) -> None:
    def _always(self, area):  # type: ignore[no-untyped-def]
        return True

    for agent in model.voting_agents:
        agent.ask_for_participation = types.MethodType(_always, agent)


def _run_one_logged_step(model, out_dir):
    logger = RunLoggerV2(
        out_dir=out_dir,
        run_seed=1,
        rule_idx=int(model.rule_idx),
        num_steps=1,
        store_grid=False,
    )
    logger.attach_to_model(model)
    logger.write_static(model)
    logger.begin_step(1)
    model.step()
    logger.log_step(step=1, model=model, grid_snapshot=None)
    logger.end_step()
    logger.finalize()

    static_payload = json.loads((out_dir / "static.json").read_text(encoding="utf-8"))
    votes = pd.read_parquet(out_dir / "votes.parquet")
    agents = pd.read_parquet(out_dir / "agents.parquet")
    return static_payload, votes, agents


def test_population_preference_knobs_interaction_end_to_end(tmp_path):
    """Interaction test for section G knobs under one-step logged execution.

    Moves several G knobs together and checks directional outcomes:
    - known_cells up => logged confidence up
    - personal_preference_peakedness up => logged personal distributions more peaked
    - num_personality_groups changes static personality-group dimensionality
    - initial_agent_assets remains reflected in logged agents output
    """
    base = dict(
        seed=1701,
        num_agents=24,
        num_areas=1,
        height=10,
        width=10,
        av_area_height=10,
        av_area_width=10,
        area_size_variance=0.0,
        num_colors=4,
        max_steps=1,
        election_cost_rate=0.0,
        reward_rate_personal=0.0,
        initial_agent_assets=100.0,
    )

    m_low, _ = create_test_model(
        **base,
        known_cells=1,
        num_personality_groups=2,
        personal_preference_peakedness=0.5,
    )
    m_high, _ = create_test_model(
        **base,
        known_cells=8,
        num_personality_groups=6,
        personal_preference_peakedness=2.0,
    )

    _force_all_participate(m_low)
    _force_all_participate(m_high)

    s_low, v_low, a_low = _run_one_logged_step(m_low, tmp_path / "low")
    s_high, v_high, a_high = _run_one_logged_step(m_high, tmp_path / "high")

    # num_personality_groups contract stays visible in static metadata
    assert len(s_low["personality_group_info"]["global_distribution"]) == 2
    assert len(s_high["personality_group_info"]["global_distribution"]) == 6

    # known_cells interaction: higher known-cells implies higher mean vote confidence
    c_low = float(v_low["confidence"].mean())
    c_high = float(v_high["confidence"].mean())
    assert c_high > c_low

    # personal_preference_peakedness interaction: higher value implies more peaked personal distributions
    p_low = np.asarray(list(s_low["personal_opt_dist"].values()), dtype=np.float64)
    p_high = np.asarray(list(s_high["personal_opt_dist"].values()), dtype=np.float64)
    assert float(np.mean(np.max(p_high, axis=1))) > float(np.mean(np.max(p_low, axis=1)))

    # initial_agent_assets survives through one logged step if economics are disabled
    np.testing.assert_allclose(float(a_low["assets"].mean()), 100.0, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(float(a_high["assets"].mean()), 100.0, rtol=0.0, atol=1e-8)
