from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import pytest

from scripts.run_headless import batch_run, run_once
from src.config.loader import load_config
from tests.factory import create_test_model


def _latest_run_root(base_out: Path) -> Path:
    roots = sorted([p for p in base_out.iterdir() if p.is_dir()])
    assert roots, f"No run root created under {base_out}"
    return roots[-1]


def test_output_directory_integration_batch_run_writes_to_configured_location(tmp_path: Path) -> None:
    cfg = load_config("toy.yaml").model_copy(deep=True)
    cfg.simulation.runs = 1
    cfg.simulation.num_steps = 1
    cfg.simulation.store_grid = False
    cfg.simulation.base_seed = 6001

    out_base = tmp_path / "custom_output"
    payload = cfg.model_dump(mode="json")
    payload["output"] = {"directory": str(out_base)}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    batch_run(str(cfg_path))
    run_root = _latest_run_root(out_base)

    assert (run_root / "config_used.yaml").exists()
    run_dir = run_root / "run_0"
    for artifact in ("meta.yaml", "static.json", "steps.parquet", "area_steps.parquet", "agents.parquet", "votes.parquet"):
        assert (run_dir / artifact).exists(), artifact


def test_schema_coverage_thesis_fields_exist_in_logged_tables(tmp_path: Path) -> None:
    cfg = load_config("toy.yaml").model_copy(deep=True)
    cfg.simulation.num_steps = 2
    cfg.simulation.store_grid = False
    cfg.simulation.base_seed = 6002
    run_dir = tmp_path / "run"
    run_once(0, cfg, out_dir=run_dir)

    steps = pd.read_parquet(run_dir / "steps.parquet")
    area_steps = pd.read_parquet(run_dir / "area_steps.parquet")
    agents = pd.read_parquet(run_dir / "agents.parquet")
    votes = pd.read_parquet(run_dir / "votes.parquet")

    for col in (
        "step",
        "turnout",
        "gini_index",
        "collective_assets",
    ):
        assert col in steps.columns
    for col in (
        "step",
        "area_id",
        "turnout",
        "gini_index",
        "participants",
        "eligible_voters",
        "dist_to_reality",
    ):
        assert col in area_steps.columns
    for col in ("step", "agent_id", "assets", "altruism_factor", "dissatisfaction_value", "participation_signal"):
        assert col in agents.columns
    for col in ("step", "area_id", "agent_id", "confidence", "rank_1_option_id", "rank_1_oppose_score"):
        assert col in votes.columns


def test_votes_logger_does_not_mask_missing_estimate_with_zeros(tmp_path: Path) -> None:
    """If voting strategy omits estimate_real_distribution, estim_dst_color_* must be NaN, not zeros."""
    model, _ = create_test_model(
        seed=6003,
        max_steps=1,
        num_areas=1,
        height=10,
        width=10,
        av_area_height=10,
        av_area_width=10,
        area_size_variance=0.0,
        num_agents=5,
        num_colors=3,
        election_cost_rate=0.0,
        reward_rate_personal=0.0,
    )

    class NoEstimateStrategy:
        def score_options(self, agent, area, options):  # type: ignore[no-untyped-def]
            # Valid score vector in [0,1], but intentionally no estimate_real_distribution call.
            n = int(options.shape[0])
            return np.full(n, 0.5, dtype=np.float32)

    for a in model.voting_agents:
        a.voting_strategy = NoEstimateStrategy()
        a.ask_for_participation = lambda area: True  # type: ignore[assignment]

    from src.logging.run_logger import RunLoggerV2

    logger = RunLoggerV2(out_dir=tmp_path / "run", run_seed=1, rule_idx=int(model.rule_idx), num_steps=1, store_grid=False)
    logger.attach_to_model(model)
    logger.write_static(model)
    logger.begin_step(1)
    model.step()
    logger.log_step(step=1, model=model, grid_snapshot=None)
    logger.end_step()
    logger.finalize()
    logger.detach_from_model(model)

    votes = pd.read_parquet(tmp_path / "run" / "votes.parquet")
    estim_cols = [c for c in votes.columns if c.startswith("estim_dst_color_")]
    assert estim_cols, "Expected expanded estim_dst_color_* columns"
    assert votes[estim_cols].isna().all(axis=None), "Missing estimates must be logged as NaN, not zeros"


def test_area_snapshot_missing_required_fields_fails_loudly(tmp_path: Path) -> None:
    model, _ = create_test_model(
        seed=6004,
        max_steps=1,
        num_areas=1,
        height=10,
        width=10,
        av_area_height=10,
        av_area_width=10,
        area_size_variance=0.0,
        num_agents=5,
        num_colors=3,
        election_cost_rate=0.0,
        reward_rate_personal=0.0,
    )
    area = model.areas[0]

    def _broken_capture() -> None:
        sink = getattr(model, "_schema_v2_area_snapshot_sink", None)
        if sink is None:
            return
        sink(
            area=area,
            snapshot={
                # intentionally omit 'participants'
                "fee_pool": 0.0,
                "eligible_voters": area.num_agents,
                "turnout": 0.0,
                "dist_to_reality": 0.0,
                "area_color": area.color_distribution.copy(),
                "elected_color": area.voted_ordering if area.voted_ordering is not None else np.array([0, 1, 2]),
            },
        )

    area._capture_area_snapshot_for_logger = _broken_capture  # type: ignore[method-assign]

    from src.logging.run_logger import RunLoggerV2

    logger = RunLoggerV2(out_dir=tmp_path / "run_broken", run_seed=1, rule_idx=int(model.rule_idx), num_steps=1, store_grid=False)
    logger.attach_to_model(model)
    logger.write_static(model)
    logger.begin_step(1)
    model.step()
    with pytest.raises(RuntimeError, match="Missing required pre-mutation snapshot fields"):
        logger.log_step(step=1, model=model, grid_snapshot=None)
    logger.end_step()
    logger.detach_from_model(model)
