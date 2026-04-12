from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_headless import run_once
from src.config.loader import load_config
from tests.factory import create_test_model


pytestmark = pytest.mark.phase1


def _dist_from_order(ordering: np.ndarray) -> np.ndarray:
    c = int(ordering.shape[0])
    arr = np.zeros(c, dtype=np.float64)
    weights = np.arange(c, 0, -1, dtype=np.float64)
    for rank, color in enumerate(ordering.tolist()):
        arr[int(color)] = float(weights[rank])
    return arr / float(np.sum(arr))


def test_quality_target_mode_validation_fail_loud() -> None:
    with pytest.raises(ValueError, match="quality_target_mode"):
        create_test_model(quality_target_mode="unsupported_mode")


def test_puzzle_process_knob_validation_fail_loud() -> None:
    with pytest.raises(ValueError, match="puzzle_local_kappa"):
        create_test_model(puzzle_local_kappa=0.0)
    with pytest.raises(ValueError, match="puzzle_local_kappa"):
        create_test_model(puzzle_local_kappa=-1.0)
    with pytest.raises(ValueError, match="puzzle_shock_prob"):
        create_test_model(puzzle_shock_prob=-0.01)
    with pytest.raises(ValueError, match="puzzle_shock_prob"):
        create_test_model(puzzle_shock_prob=1.01)


def test_puzzle_distribution_shock_endpoints_and_prev_none_oracle() -> None:
    model, _ = create_test_model(seed=9301, quality_target_mode="puzzle")
    area = model.areas[0]
    prev = np.asarray([0.7, 0.2, 0.1], dtype=np.float64)
    fresh = np.asarray([0.2, 0.3, 0.5], dtype=np.float64)
    local = np.asarray([0.6, 0.2, 0.2], dtype=np.float64)
    area._sample_fresh_puzzle_distribution = lambda: fresh.copy()  # type: ignore[method-assign]
    area._sample_local_puzzle_distribution = lambda p: local.copy()  # type: ignore[method-assign]

    # No previous puzzle => always fresh, independent of shock probability.
    area._puzzle_distribution = None
    model.puzzle_shock_prob = 0.0
    area._update_puzzle_distribution()
    np.testing.assert_allclose(area.puzzle_distribution, fresh, rtol=0.0, atol=1e-12)

    # shock_prob=0 => local step when previous puzzle exists.
    area._puzzle_distribution = prev.copy()
    model.puzzle_shock_prob = 0.0
    area._update_puzzle_distribution()
    np.testing.assert_allclose(area.puzzle_distribution, local, rtol=0.0, atol=1e-12)

    # shock_prob=1 => always fresh.
    area._puzzle_distribution = prev.copy()
    model.puzzle_shock_prob = 1.0
    area._update_puzzle_distribution()
    np.testing.assert_allclose(area.puzzle_distribution, fresh, rtol=0.0, atol=1e-12)


def test_puzzle_local_dirichlet_mean_is_centered_on_previous_distribution() -> None:
    model, _ = create_test_model(seed=9304, quality_target_mode="puzzle", num_colors=4)
    area = model.areas[0]
    model.puzzle_local_kappa = 25.0
    prev = np.asarray([0.50, 0.20, 0.20, 0.10], dtype=np.float64)
    alpha_center_pc = 2.0

    draws = np.asarray([area._sample_local_puzzle_distribution(prev) for _ in range(600)], dtype=np.float64)
    assert draws.shape == (600, 4)
    total = float(model.puzzle_local_kappa) + float(model.num_colors) * alpha_center_pc
    expected = (float(model.puzzle_local_kappa) * prev + alpha_center_pc) / total
    np.testing.assert_allclose(np.mean(draws, axis=0), expected, atol=0.035, rtol=0.0)


def test_puzzle_local_kappa_controls_mean_jump_size() -> None:
    prev = np.asarray([0.50, 0.20, 0.20, 0.10], dtype=np.float64)

    m_low, _ = create_test_model(seed=9305, quality_target_mode="puzzle", num_colors=4)
    m_high, _ = create_test_model(seed=9305, quality_target_mode="puzzle", num_colors=4)
    a_low = m_low.areas[0]
    a_high = m_high.areas[0]
    m_low.puzzle_local_kappa = 3.0
    m_high.puzzle_local_kappa = 80.0

    jumps_low = []
    jumps_high = []
    for _ in range(400):
        d_low = a_low._sample_local_puzzle_distribution(prev)
        d_high = a_high._sample_local_puzzle_distribution(prev)
        jumps_low.append(float(np.sum(np.abs(d_low - prev))))
        jumps_high.append(float(np.sum(np.abs(d_high - prev))))

    assert float(np.mean(jumps_high)) < float(np.mean(jumps_low))


def test_ordering_from_distribution_tie_aware_returns_permutation_for_near_equal_values() -> None:
    dist = np.asarray([0.5, 0.25, 0.25 + 5e-13, 0.0], dtype=np.float64)
    ordering = type(create_test_model(seed=9306)[0].areas[0])._ordering_from_distribution_tie_aware(
        dist,
        reference_ordering=np.asarray([0, 2, 1, 3], dtype=np.int64),
        rng=None,
    )
    assert sorted(ordering.tolist()) == [0, 1, 2, 3]


def test_puzzle_quality_gate_switch_changes_reward_sign_path() -> None:
    common = dict(
        seed=9302,
        num_colors=3,
        num_agents=8,
        num_areas=1,
        election_cost_rate=0.0,
        reward_rate_personal=0.10,
        break_even_distance_common=0.25,
        max_steps=1,
    )
    m_reality, _ = create_test_model(**common, quality_target_mode="reality")
    m_puzzle, _ = create_test_model(**common, quality_target_mode="puzzle")
    a_reality = m_reality.global_area
    a_puzzle = m_puzzle.global_area
    assert int(a_reality.num_agents) > 0
    assert int(a_puzzle.num_agents) > 0

    # Force same elected ordering in both worlds via direct area state.
    elected = np.asarray(a_reality.agents[0].personality_group, dtype=np.int64)
    a_reality._voted_ordering = elected.copy()
    a_puzzle._voted_ordering = elected.copy()

    # Reality branch should be "bad", puzzle branch should be "good".
    a_reality._color_distribution = _dist_from_order(elected[::-1])
    a_puzzle._color_distribution = _dist_from_order(elected[::-1])
    a_reality._puzzle_distribution = _dist_from_order(elected)
    a_puzzle._puzzle_distribution = _dist_from_order(elected)

    for agent in a_reality.agents:
        agent.reset_reward_variables()
    for agent in a_puzzle.agents:
        agent.reset_reward_variables()

    tracked_reality = a_reality.agents[0]
    tracked_puzzle = a_puzzle.agents[0]

    a_reality._distribute_rewards()
    a_puzzle._distribute_rewards()

    # Same personality group as elected ordering:
    # reality "bad" => reward factor 0 => no reward
    assert float(tracked_reality.reward_personal) == pytest.approx(0.0, abs=1e-10)
    # puzzle "good" => full positive reward
    assert float(tracked_puzzle.reward_personal) > 0.0
    assert float(a_reality.dist_to_reality) > float(m_reality.break_even_distance_common)
    assert float(a_puzzle.puzzle_distance) <= float(m_puzzle.break_even_distance_common) + 1e-12


def test_puzzle_mode_known_cells_sampling_uses_color_samples() -> None:
    model, _ = create_test_model(
        seed=9303,
        quality_target_mode="puzzle",
        known_cells=20,
        num_colors=3,
        num_agents=2,
        num_areas=1,
    )
    area = model.areas[0]
    agent = model.voting_agents[0]
    area._puzzle_distribution = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)

    agent.update_known_cells(area=area)
    assert len(agent.known_cells) == int(model.known_cells)
    assert all(isinstance(x, int) for x in agent.known_cells)
    assert all(int(x) == 0 for x in agent.known_cells)

    est, conf = agent.estimate_real_distribution(area=area)
    np.testing.assert_allclose(est, np.asarray([1.0, 0.0, 0.0], dtype=np.float64), rtol=0.0, atol=1e-12)
    assert conf > 0.0


def test_puzzle_mode_logs_puzzle_distribution_vectors_in_area_steps(tmp_path: Path) -> None:
    cfg = load_config("toy.yaml")
    cfg_run = cfg.model_copy(deep=True)
    cfg_run.simulation.num_steps = 2
    cfg_run.simulation.runs = 1
    cfg_run.simulation.store_grid = False
    cfg_run.model.quality_target_mode = "puzzle"
    cfg_run.model.altruism_mode = "satisfaction"
    out = tmp_path / "run"
    run_once(0, cfg_run, out_dir=out)

    area_steps = pd.read_parquet(out / "area_steps.parquet").sort_values(["step", "area_id"]).reset_index(drop=True)
    puzzle_cols = [c for c in area_steps.columns if c.startswith("puzzle_color_")]
    assert puzzle_cols, "Expected puzzle_color_* columns in area_steps.parquet for puzzle-mode run"
    assert "grid_ordering_id" in area_steps.columns
    assert "puzzle_ordering_id" in area_steps.columns
    assert np.isfinite(area_steps["grid_ordering_id"].to_numpy(dtype=float)).all()
    assert np.isfinite(area_steps["puzzle_ordering_id"].to_numpy(dtype=float)).all()
    vals = area_steps[puzzle_cols].to_numpy(dtype=float)
    assert np.all(np.isfinite(vals))
    np.testing.assert_allclose(np.sum(vals, axis=1), np.ones(vals.shape[0]), atol=1e-5, rtol=0.0)


def test_puzzle_distribution_logging_avoids_degenerate_zero_components_in_local_rw(tmp_path: Path) -> None:
    cfg = load_config("toy.yaml")
    cfg_run = cfg.model_copy(deep=True)
    cfg_run.simulation.num_steps = 40
    cfg_run.simulation.runs = 1
    cfg_run.simulation.store_grid = False
    cfg_run.model.quality_target_mode = "puzzle"
    cfg_run.model.altruism_mode = "satisfaction"
    cfg_run.model.puzzle_shock_prob = 0.0
    cfg_run.model.puzzle_local_kappa = 5.0
    out = tmp_path / "run"
    run_once(0, cfg_run, out_dir=out)

    area_steps = pd.read_parquet(out / "area_steps.parquet")
    puzzle_cols = [c for c in area_steps.columns if c.startswith("puzzle_color_")]
    vals = area_steps[puzzle_cols].to_numpy(dtype=float)
    assert vals.size > 0
    assert not np.any(vals == 0.0), "Puzzle generator/logging still produces exact zero components"
    # Center-biased Dirichlet priors should keep puzzle components away from simplex walls.
    assert float(vals.min()) > 1e-4, "Puzzle generator still hugs the simplex wall too closely"


def test_puzzle_time_series_reproducible_for_same_seed_and_config(tmp_path: Path) -> None:
    cfg = load_config("toy.yaml").model_copy(deep=True)
    cfg.simulation.num_steps = 6
    cfg.simulation.runs = 1
    cfg.simulation.store_grid = False
    cfg.model.quality_target_mode = "puzzle"
    cfg.model.altruism_mode = "satisfaction"
    cfg.simulation.base_seed = 9410

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    run_once(0, cfg, out_dir=out_a)
    run_once(0, cfg, out_dir=out_b)

    def _puzzle_cols(out_dir: Path) -> pd.DataFrame:
        area_steps = pd.read_parquet(out_dir / "area_steps.parquet").sort_values(["step", "area_id"]).reset_index(drop=True)
        cols = [c for c in area_steps.columns if c.startswith("puzzle_color_")]
        assert cols
        return area_steps[cols]

    pd.testing.assert_frame_equal(_puzzle_cols(out_a), _puzzle_cols(out_b), check_dtype=False)


def test_puzzle_time_series_reproducible_across_rule_idx_for_same_seed(tmp_path: Path) -> None:
    cfg = load_config("toy.yaml").model_copy(deep=True)
    cfg.simulation.num_steps = 6
    cfg.simulation.runs = 1
    cfg.simulation.store_grid = False
    cfg.model.quality_target_mode = "puzzle"
    cfg.model.altruism_mode = "satisfaction"
    cfg.simulation.base_seed = 9411

    cfg_a = cfg.model_copy(deep=True)
    cfg_b = cfg.model_copy(deep=True)
    cfg_a.model.rule_idx = 1  # approval
    cfg_b.model.rule_idx = 0  # plurality

    out_a = tmp_path / "approval"
    out_b = tmp_path / "plurality"
    run_once(0, cfg_a, out_dir=out_a)
    run_once(0, cfg_b, out_dir=out_b)

    def _puzzle_cols(out_dir: Path) -> pd.DataFrame:
        area_steps = pd.read_parquet(out_dir / "area_steps.parquet").sort_values(["step", "area_id"]).reset_index(drop=True)
        cols = [c for c in area_steps.columns if c.startswith("puzzle_color_")]
        assert cols
        return area_steps[cols]

    pd.testing.assert_frame_equal(_puzzle_cols(out_a), _puzzle_cols(out_b), check_dtype=False)
