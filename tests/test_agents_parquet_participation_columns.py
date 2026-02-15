from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.logging.run_logger import RunLoggerV2
from tests.factory import create_test_model


def test_agents_parquet_includes_participation_baseline_columns(tmp_path: Path) -> None:
    model, _ = create_test_model(num_agents=2, num_areas=1, num_colors=3)

    logger = RunLoggerV2(
        out_dir=tmp_path,
        run_seed=1,
        rule_idx=0,
        num_steps=1,
        store_grid=False,
    )

    # Minimal area snapshot so logging can proceed without missing pre-mutation data.
    step = 1
    elected_color = model.options[0]
    area0 = model.areas[0]
    agent0 = model.voting_agents[0]
    for area in model.areas:
        area._dist_to_reality = 0.0
        logger._area_snapshots_by_step_area[(step, int(area.unique_id))] = {
            "area_color": area.color_distribution.copy(),
            "elected_color": elected_color,
            "eligible_voters": int(area.num_agents),
            "participants": 0,
            "turnout": float(area.voter_turnout),
            "election_cost_rate": float(model.election_cost_rate),
            "fee_pool": float(getattr(area, "_election_fee_pool", 0.0)),
            "gini_index": 0,
            "dist_to_reality": 0.0,
        }

    logger._votes_rows.append(
        {
            "run_seed": np.int32(1),
            "rule_idx": np.int16(0),
            "step": np.int32(step),
            "area_id": np.int32(area0.unique_id),
            "agent_id": np.int32(agent0.unique_id),
            "participating": True,
            "confidence": np.float32(1.0),
            "rank_1_option_id": np.int32(0),
            "rank_1_oppose_score": np.float32(0.0),
            "rank_2_option_id": np.int32(1),
            "rank_2_oppose_score": np.float32(0.0),
            "rank_3_option_id": np.int32(2),
            "rank_3_oppose_score": np.float32(0.0),
            "estim_dst_color_0": np.float32(0.34),
            "estim_dst_color_1": np.float32(0.33),
            "estim_dst_color_2": np.float32(0.33),
        }
    )

    logger.log_step(step=step, model=model, grid_snapshot=None)
    logger.finalize()

    agents_df = pd.read_parquet(tmp_path / "agents.parquet")
    assert "participation_baseline" in agents_df.columns
    assert "participation_signal" in agents_df.columns
