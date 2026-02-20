from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_headless import run_once
from src.config.loader import load_config
from src.utils.representations import validate_ordering
from tests.factory import create_test_model


pytestmark = pytest.mark.phase1


def _assert_distribution(x: np.ndarray, n: int, *, atol: float = 1e-6) -> None:
    arr = np.asarray(x, dtype=np.float64)
    assert arr.shape == (n,)
    assert np.all(np.isfinite(arr))
    assert float(arr.min(initial=0.0)) >= -atol
    assert abs(float(arr.sum()) - 1.0) <= atol


def _grid_distribution(model) -> np.ndarray:
    counts = np.zeros(int(model.num_colors), dtype=np.float64)
    for cell in model.color_cells:
        counts[int(cell.color)] += 1.0
    return counts / float(len(model.color_cells))


def _assert_runtime_invariants(model) -> None:
    # Global distribution must always match the realized grid.
    _assert_distribution(np.asarray(model.global_color_dst, dtype=np.float64), int(model.num_colors))
    np.testing.assert_allclose(
        np.asarray(model.global_color_dst, dtype=np.float64),
        _grid_distribution(model),
        rtol=0.0,
        atol=1e-8,
    )

    for area in model.areas:
        assert area.num_cells >= 1
        assert len(area.cells) == int(area.num_cells)
        assert 0 <= int(area.num_agents_participated_last) <= int(area.num_agents)
        assert 0.0 <= float(area.voter_turnout) <= 100.0
        if area.dist_to_reality is not None:
            assert np.isfinite(float(area.dist_to_reality))
            assert 0.0 <= float(area.dist_to_reality) <= 1.0 + 1e-6
        if area.voted_ordering is not None:
            validate_ordering(np.asarray(area.voted_ordering), int(model.num_colors))

        _assert_distribution(np.asarray(area.color_distribution, dtype=np.float64), int(model.num_colors))
        counts = np.asarray(area.color_counts, dtype=np.int64)
        assert counts.shape == (int(model.num_colors),)
        assert int(np.sum(counts)) == int(area.num_cells)
        assert np.all(counts >= 0)
        np.testing.assert_allclose(
            np.asarray(area.color_distribution, dtype=np.float64),
            counts.astype(np.float64) / float(area.num_cells),
            rtol=0.0,
            atol=1e-8,
        )

    for agent in model.voting_agents:
        assert np.isfinite(float(agent.assets))
        assert float(agent.assets) >= 0.0
        assert np.isfinite(float(agent.q_participation))
        assert np.isfinite(float(agent.participation_signal))
        assert np.isfinite(float(agent.altruism_factor))
        assert np.isfinite(float(agent.dissatisfaction_value))
        assert np.isfinite(float(agent.dissatisfaction_signal))
        assert np.isfinite(float(agent.election_delta_abs))
        assert np.isfinite(float(agent.election_delta_rel))
        p = float(agent.participation_probability())
        assert np.isfinite(p)
        assert 0.0 <= p <= 1.0

        q_max = float(model.participation_q_max)
        assert -q_max - 1e-12 <= float(agent.q_participation) <= q_max + 1e-12

        if model.altruism_learning:
            assert float(model.altruism_clip_min) - 1e-12 <= float(agent.altruism_factor)
            assert float(agent.altruism_factor) <= float(model.altruism_clip_max) + 1e-12
        else:
            np.testing.assert_allclose(
                float(agent.altruism_factor),
                float(model.altruism_static),
                rtol=0.0,
                atol=1e-12,
            )


@pytest.mark.parametrize(
    "overrides",
    [
        # Baseline disjoint.
        dict(num_colors=4, num_agents=50, num_areas=4, height=10, width=10, av_area_height=5, av_area_width=5, area_size_variance=0.0),
        # Overlap-heavy geometry.
        dict(num_colors=4, num_agents=50, num_areas=3, height=10, width=10, av_area_height=8, av_area_width=8, area_size_variance=0.35),
        # Strong economics.
        dict(
            num_colors=4,
            num_agents=50,
            num_areas=4,
            height=10,
            width=10,
            av_area_height=5,
            av_area_width=5,
            area_size_variance=0.0,
            election_cost_rate=0.3,
            reward_rate_personal=0.35,
            break_even_distance_common=0.25,
        ),
        # Strong learning dynamics.
        dict(
            num_colors=4,
            num_agents=50,
            num_areas=4,
            height=10,
            width=10,
            av_area_height=5,
            av_area_width=5,
            area_size_variance=0.0,
            participation_alpha=0.8,
            participation_beta=8.0,
            participation_q_max=2.0,
            participation_baseline_alpha=0.4,
            altruism_learning=True,
            altruism_alpha=0.6,
            altruism_init=0.5,
            satisfaction_baseline_alpha=0.4,
        ),
        # Extreme mutation.
        dict(
            num_colors=4,
            num_agents=50,
            num_areas=4,
            height=10,
            width=10,
            av_area_height=5,
            av_area_width=5,
            area_size_variance=0.0,
            mu=1.0,
            election_impact_on_mutation=3.0,
        ),
    ],
)
def test_runtime_invariants_hold_under_adversarial_seed_sweep(overrides: dict) -> None:
    for seed in (1101, 1102, 1103):
        model, _ = create_test_model(seed=seed, max_steps=9, **overrides)
        for _ in range(8):
            model.step()
            _assert_runtime_invariants(model)


@pytest.mark.parametrize(
    "label, model_overrides",
    [
        (
            "baseline",
            dict(
                num_colors=4,
                num_agents=40,
                num_areas=4,
                height=10,
                width=10,
                av_area_height=5,
                av_area_width=5,
                area_size_variance=0.0,
            ),
        ),
        (
            "stress",
            dict(
                num_colors=4,
                num_agents=40,
                num_areas=3,
                height=10,
                width=10,
                av_area_height=8,
                av_area_width=8,
                area_size_variance=0.35,
                election_cost_rate=0.25,
                reward_rate_personal=0.3,
                break_even_distance_common=0.2,
                mu=0.8,
            ),
        ),
    ],
)
def test_headless_logged_series_invariants_under_adversarial_configs(
    tmp_path: Path,
    label: str,
    model_overrides: dict,
) -> None:
    cfg = load_config("toy.yaml").model_copy(deep=True)
    cfg.simulation.num_steps = 6
    cfg.simulation.store_grid = False
    cfg.simulation.base_seed = 5200
    cfg.simulation.runs = 1
    for k, v in model_overrides.items():
        setattr(cfg.model, k, v)

    run_dir = tmp_path / label
    run_once(run_id=0, cfg=cfg, out_dir=run_dir)

    steps = pd.read_parquet(run_dir / "steps.parquet")
    area_steps = pd.read_parquet(run_dir / "area_steps.parquet")
    votes = pd.read_parquet(run_dir / "votes.parquet")

    assert len(steps) == int(cfg.simulation.num_steps)
    assert np.all(np.isfinite(steps["collective_assets"].to_numpy(dtype=float)))
    assert np.all(np.isfinite(steps["turnout"].to_numpy(dtype=float)))
    assert np.all(np.isfinite(steps["gini_index"].to_numpy(dtype=float)))
    assert np.all((steps["turnout"].to_numpy(dtype=float) >= 0.0) & (steps["turnout"].to_numpy(dtype=float) <= 100.0))
    # Current schema/unit contract: gini_index is stored as percent-like 0..100.
    assert np.all((steps["gini_index"].to_numpy(dtype=float) >= 0.0) & (steps["gini_index"].to_numpy(dtype=float) <= 100.0 + 1e-6))

    step_color_cols = sorted(c for c in steps.columns if c.startswith("color_"))
    assert step_color_cols
    step_color_sums = steps[step_color_cols].sum(axis=1).to_numpy(dtype=float)
    np.testing.assert_allclose(step_color_sums, np.ones_like(step_color_sums), rtol=0.0, atol=1e-6)

    assert np.all(np.isfinite(area_steps["turnout"].to_numpy(dtype=float)))
    assert np.all(np.isfinite(area_steps["dist_to_reality"].to_numpy(dtype=float)))
    assert np.all((area_steps["turnout"].to_numpy(dtype=float) >= 0.0) & (area_steps["turnout"].to_numpy(dtype=float) <= 100.0))
    assert np.all((area_steps["dist_to_reality"].to_numpy(dtype=float) >= 0.0) & (area_steps["dist_to_reality"].to_numpy(dtype=float) <= 1.0 + 1e-6))

    area_color_cols = sorted(c for c in area_steps.columns if c.startswith("area_color_"))
    assert area_color_cols
    area_color_sums = area_steps[area_color_cols].sum(axis=1).to_numpy(dtype=float)
    np.testing.assert_allclose(area_color_sums, np.ones_like(area_color_sums), rtol=0.0, atol=1e-6)

    # Vote rows may include NaN estimates (intended when estimate is missing), but if present they
    # must stay in [0,1] and sum close to one.
    estim_cols = sorted(c for c in votes.columns if c.startswith("estim_dst_color_"))
    if estim_cols:
        est = votes[estim_cols].to_numpy(dtype=float)
        finite_mask = np.isfinite(est).all(axis=1)
        if np.any(finite_mask):
            est_fin = est[finite_mask]
            assert np.all(est_fin >= -1e-6)
            assert np.all(est_fin <= 1.0 + 1e-6)
            np.testing.assert_allclose(est_fin.sum(axis=1), np.ones(est_fin.shape[0]), rtol=0.0, atol=1e-5)
