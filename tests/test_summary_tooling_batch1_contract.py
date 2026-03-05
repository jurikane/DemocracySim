from __future__ import annotations

from math import factorial, log
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from scripts.run_headless import run_once
from src.analysis.thesis_endpoints import step_volatility_l1_normalized
from src.analysis.summary_tooling import generate_run_summary_batch1
from src.config.loader import load_config
from src.utils.metrics import gini_index_0_100


def _make_small_run(tmp_path: Path) -> Path:
    cfg = load_config("toy.yaml").model_copy(deep=True)
    cfg.simulation.num_steps = 10
    cfg.simulation.base_seed = 17001
    cfg.simulation.store_grid = False
    out = tmp_path / "run_0"
    run_once(run_id=0, cfg=cfg, out_dir=out)
    return out


def _expected_global_dist_to_reality(area_steps: pd.DataFrame, step: int) -> float:
    block = area_steps[area_steps["step"] == step]
    weights = block["eligible_voters"].to_numpy(dtype=float)
    vals = block["dist_to_reality"].to_numpy(dtype=float)
    denom = float(np.sum(weights))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(vals * weights) / denom)


def _expected_global_diversity(votes: pd.DataFrame, *, step: int, num_colors: int) -> float:
    block = votes[votes["step"] == step]
    if block.empty:
        return float("nan")
    s = block["rank_1_option_id"].dropna()
    if s.empty:
        return float("nan")
    counts = s.value_counts().to_numpy(dtype=float)
    probs = counts / float(np.sum(counts))
    entropy = float(-np.sum(probs * np.log(probs + 1e-15)))
    norm = log(float(factorial(num_colors)))
    return float(max(0.0, min(1.0, entropy / norm)))


def test_batch1_summary_generation_writes_core_artifacts_and_columns(tmp_path: Path) -> None:
    run_dir = _make_small_run(tmp_path)

    artifacts = generate_run_summary_batch1(run_dir=run_dir)
    assert artifacts.global_series_csv.exists()
    assert artifacts.area_series_csv.exists()
    assert artifacts.summary_stats_json.exists()

    global_df = pd.read_csv(artifacts.global_series_csv)
    area_df = pd.read_csv(artifacts.area_series_csv)
    stats = json.loads(artifacts.summary_stats_json.read_text(encoding="utf-8"))
    global_summary = stats["global_summary"]

    global_required = {
        "step",
        "turnout",
        "gini_assets",
        "gini_dissatisfaction",
        "mean_dissatisfaction",
        "dist_to_reality",
        "dist_to_ref_utilitarian",
        "dist_to_ref_nash",
        "dist_to_ref_egalitarian",
        "dist_to_ref_rawlsian",
        "dist_to_ref_egalitarian_lam025",
        "dist_to_ref_egalitarian_lam400",
        "diversity_first_choice_entropy",
    }
    assert global_required.issubset(set(global_df.columns))
    assert np.array_equal(
        global_df["step"].to_numpy(dtype=int),
        np.arange(1, len(global_df) + 1, dtype=int),
    )

    area_required = {
        "step",
        "area_id",
        "participants",
        "eligible_voters",
        "turnout",
        "gini_assets",
        "dist_to_reality",
        "dist_to_ref_utilitarian",
        "dist_to_ref_nash",
        "dist_to_ref_egalitarian",
        "dist_to_ref_rawlsian",
        "dist_to_ref_egalitarian_lam025",
        "dist_to_ref_egalitarian_lam400",
        "diversity_first_choice_entropy",
        "altruistic_rank1_match_puzzle_share",
        "non_altruistic_rank1_match_puzzle_share",
        "altruistic_rank1_match_outcome_share",
        "non_altruistic_rank1_match_outcome_share",
        "altruistic_votes_count",
        "non_altruistic_votes_count",
        "vote_count_total",
        "altruistic_vote_share",
        "non_altruistic_vote_share",
    }
    assert area_required.issubset(set(area_df.columns))
    assert stats["shape"]["num_steps"] == len(global_df)
    assert stats["shape"]["num_areas"] == int(area_df["area_id"].nunique())
    for key in (
        "turnout_volatility",
        "gini_assets_volatility",
        "gini_dissatisfaction_volatility",
        "dist_to_reality_volatility",
    ):
        assert key in global_summary


def test_batch1_summary_formulas_match_logged_artifacts(tmp_path: Path) -> None:
    run_dir = _make_small_run(tmp_path)
    artifacts = generate_run_summary_batch1(run_dir=run_dir)

    steps = pd.read_parquet(run_dir / "steps.parquet").sort_values("step").reset_index(drop=True)
    area_steps = pd.read_parquet(run_dir / "area_steps.parquet").sort_values(["step", "area_id"]).reset_index(drop=True)
    agents = pd.read_parquet(run_dir / "agents.parquet").sort_values(["step", "agent_id"]).reset_index(drop=True)
    votes = pd.read_parquet(run_dir / "votes.parquet").sort_values(["step", "area_id", "agent_id"]).reset_index(drop=True)
    static = json.loads((run_dir / "static.json").read_text(encoding="utf-8"))
    num_colors = int(static["num_colors"])

    summary_global = pd.read_csv(artifacts.global_series_csv).sort_values("step").reset_index(drop=True)

    # gini_dissatisfaction from agents.dissatisfaction_value
    expected_gini = (
        agents.groupby("step", sort=True)["dissatisfaction_value"]
        .apply(lambda s: float(gini_index_0_100(s.to_numpy(dtype=float))))
        .reindex(summary_global["step"].to_numpy(dtype=int), fill_value=np.nan)
        .to_numpy(dtype=float)
    )
    np.testing.assert_allclose(
        summary_global["gini_dissatisfaction"].to_numpy(dtype=float),
        expected_gini,
        rtol=0.0,
        atol=1e-6,
        equal_nan=True,
    )

    # dist_to_reality weighted by eligible_voters
    expected_dist = np.asarray(
        [
            _expected_global_dist_to_reality(area_steps, int(step))
            for step in summary_global["step"].to_numpy(dtype=int)
        ],
        dtype=float,
    )
    np.testing.assert_allclose(
        summary_global["dist_to_reality"].to_numpy(dtype=float),
        expected_dist,
        rtol=0.0,
        atol=1e-6,
        equal_nan=True,
    )

    # diversity entropy from votes.rank_1_option_id
    expected_div = np.asarray(
        [
            _expected_global_diversity(votes, step=int(step), num_colors=num_colors)
            for step in summary_global["step"].to_numpy(dtype=int)
        ],
        dtype=float,
    )
    actual_div = summary_global["diversity_first_choice_entropy"].to_numpy(dtype=float)
    np.testing.assert_allclose(actual_div, expected_div, rtol=0.0, atol=1e-6, equal_nan=True)
    finite = actual_div[np.isfinite(actual_div)]
    assert np.all((finite >= 0.0) & (finite <= 1.0))

    # passthrough columns remain aligned with steps.parquet
    np.testing.assert_allclose(
        summary_global["turnout"].to_numpy(dtype=float),
        steps["turnout"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        summary_global["gini_assets"].to_numpy(dtype=float),
        steps["gini_index"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-6,
    )

    stats = json.loads(artifacts.summary_stats_json.read_text(encoding="utf-8"))
    gs = stats["global_summary"]
    assert float(gs["turnout_volatility"]) == pytest.approx(
        step_volatility_l1_normalized(summary_global["turnout"].to_numpy(dtype=float), value_range=100.0),
        abs=1e-6,
    )
    assert float(gs["gini_assets_volatility"]) == pytest.approx(
        step_volatility_l1_normalized(summary_global["gini_assets"].to_numpy(dtype=float), value_range=100.0),
        abs=1e-6,
    )
    assert float(gs["gini_dissatisfaction_volatility"]) == pytest.approx(
        step_volatility_l1_normalized(summary_global["gini_dissatisfaction"].to_numpy(dtype=float), value_range=100.0),
        abs=1e-6,
    )
    assert float(gs["dist_to_reality_volatility"]) == pytest.approx(
        step_volatility_l1_normalized(summary_global["dist_to_reality"].to_numpy(dtype=float), value_range=1.0),
        abs=1e-6,
    )
