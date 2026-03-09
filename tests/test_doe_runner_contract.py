from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from src.analysis.doe_runner import (
    apply_doe_overrides,
    build_run_plan,
    get_doe_profile,
    midpoint_params_from_ranges,
    rule_label,
    sample_design_points,
    select_farthest_seeds_from_descriptors,
    select_stratified_seeds,
    write_design_manifest,
    write_run_manifest,
    write_seed_selection_manifest,
)
from src.config.loader import load_config


def test_sample_design_points_respects_rate_sum_constraint() -> None:
    rng = np.random.default_rng(123)
    points = sample_design_points(num_points=40, rng=rng)
    assert len(points) == 40
    for p in points:
        assert "participation_baseline_alpha" not in p
        s = float(p["election_cost_rate"] + p["reward_rate_personal"])
        assert s <= 0.9 + 1e-12


def test_build_run_plan_primary_plus_robust_every_2(tmp_path: Path) -> None:
    points = [
        {"election_cost_rate": 0.01, "reward_rate_personal": 0.03, "break_even_distance_common": 0.2, "election_impact_on_mutation": 1.0, "mu": 0.3, "participation_alpha": 0.05, "participation_beta": 1.0, "participation_init_q": 0.0, "altruism_static": 0.5},
        {"election_cost_rate": 0.01, "reward_rate_personal": 0.03, "break_even_distance_common": 0.2, "election_impact_on_mutation": 1.0, "mu": 0.3, "participation_alpha": 0.05, "participation_beta": 1.0, "participation_init_q": 0.0, "altruism_static": 0.5},
        {"election_cost_rate": 0.01, "reward_rate_personal": 0.03, "break_even_distance_common": 0.2, "election_impact_on_mutation": 1.0, "mu": 0.3, "participation_alpha": 0.05, "participation_beta": 1.0, "participation_init_q": 0.0, "altruism_static": 0.5},
    ]
    seeds = [101, 202]
    plan = build_run_plan(
        out_root=tmp_path,
        design_points=points,
        seeds=seeds,
        primary_rule_idx=1,
        robust_rule_idx=2,
        include_robustness=True,
        robust_every=2,
    )
    # Primary always: 3*2 = 6; robustness on design 0 and 2: 2*2 = 4.
    assert len(plan) == 10
    assert sum(1 for t in plan if t.rule_idx == 1) == 6
    assert sum(1 for t in plan if t.rule_idx == 2) == 4


def test_apply_doe_overrides_sets_frozen_and_run_controls() -> None:
    cfg = load_config("toy.yaml")
    params = {
        "election_cost_rate": 0.07,
        "reward_rate_personal": 0.04,
        "break_even_distance_common": 0.25,
        "election_impact_on_mutation": 2.5,
        "mu": 0.8,
        "participation_alpha": 0.1,
        "participation_beta": 3.0,
        "participation_init_q": -0.2,
        "altruism_static": 0.4,
    }
    out = apply_doe_overrides(cfg, params=params, rule_idx=2, base_seed=303)
    assert out.model.rule_idx == 2
    assert out.simulation.base_seed == 303
    assert out.simulation.store_grid is False
    assert out.model.num_areas == 1
    assert abs(out.model.election_cost_rate - 0.07) < 1e-12
    assert abs(out.model.altruism_static - 0.4) < 1e-12
    assert out.model.participation_q_max == 2.0
    assert out.model.participation_baseline_alpha == 0.1
    assert out.model.color_patches_steps == 0


def test_write_design_manifest_writes_spec_and_csv(tmp_path: Path) -> None:
    points = [
        {
            "election_cost_rate": 0.01,
            "reward_rate_personal": 0.03,
            "break_even_distance_common": 0.2,
            "election_impact_on_mutation": 1.0,
            "mu": 0.3,
            "participation_alpha": 0.05,
            "participation_beta": 1.0,
            "participation_init_q": 0.0,
            "altruism_static": 0.5,
        }
    ]
    write_design_manifest(
        out_root=tmp_path,
        design_points=points,
        seeds=[101, 202, 303],
        primary_rule_idx=1,
        robust_rule_idx=2,
        include_robustness=True,
        robust_every=1,
    )
    assert (tmp_path / "doe_spec.json").exists()
    assert (tmp_path / "doe_design_points.csv").exists()
    spec = json.loads((tmp_path / "doe_spec.json").read_text(encoding="utf-8"))
    assert spec["profile"] == "phase1"
    assert spec["primary_rule_name"] == "approval"
    assert spec["robust_rule_name"] == "utilitarian"


def test_get_doe_profile_phase2_has_altruism_learning_and_ranges() -> None:
    p = get_doe_profile("phase2_altruism_learning")
    assert p["frozen_model"]["altruism_learning"] is True
    assert "altruism_alpha" in p["ranges"]
    assert "altruism_init" in p["ranges"]
    assert "altruism_static" not in p["ranges"]


def test_get_doe_profile_phase2_probe_includes_known_cells_and_focuses_altruism() -> None:
    p = get_doe_profile("phase2_altruism_probe")
    assert p["frozen_model"]["altruism_learning"] is True
    assert {
        "altruism_alpha",
        "altruism_init",
        "satisfaction_baseline_alpha",
        "known_cells",
    }.issubset(set(p["ranges"].keys()))
    assert p["ranges"]["known_cells"] == (float(p["frozen_model"]["num_colors"]), 30.0)


def test_get_doe_profile_phase3_puzzle_main_includes_known_cells_and_altruism_learning() -> None:
    p = get_doe_profile("phase3_puzzle_main")
    assert p["frozen_model"]["quality_target_mode"] == "puzzle"
    assert p["frozen_model"]["altruism_mode"] == "satisfaction"
    assert p["frozen_model"]["altruism_response_gamma"] == 1.0
    assert {
        "known_cells",
        "altruism_response_gamma",
        "puzzle_local_kappa",
        "puzzle_shock_prob",
    }.issubset(set(p["ranges"].keys()))
    assert p["ranges"]["known_cells"] == (float(p["frozen_model"]["num_colors"]), 30.0)


def test_sample_design_points_known_cells_is_integer_valued() -> None:
    p = get_doe_profile("phase2_altruism_probe")
    rng = np.random.default_rng(7)
    points = sample_design_points(num_points=30, rng=rng, ranges=p["ranges"])
    known_values = [pt["known_cells"] for pt in points]
    assert all(float(v).is_integer() for v in known_values)
    assert min(known_values) >= 4.0
    assert max(known_values) <= 30.0


def test_midpoint_params_from_ranges() -> None:
    ranges = {
        "a": (0.0, 2.0),
        "b": (-1.0, 1.0),
    }
    p = midpoint_params_from_ranges(ranges)
    assert p["a"] == 1.0
    assert p["b"] == 0.0


def test_select_farthest_seeds_from_descriptors() -> None:
    descriptors = {
        10: np.array([0.0, 0.0], dtype=float),
        11: np.array([0.1, 0.0], dtype=float),
        12: np.array([10.0, 10.0], dtype=float),
        13: np.array([9.9, 10.0], dtype=float),
    }
    selected = select_farthest_seeds_from_descriptors(descriptors, target_count=2)
    assert len(selected) == 2
    assert 12 in selected or 13 in selected
    assert 10 in selected or 11 in selected


def test_select_stratified_seeds_returns_subset() -> None:
    cfg = load_config("toy.yaml")
    candidates = [101, 202, 303, 404]
    selected = select_stratified_seeds(
        cfg,
        target_count=2,
        candidate_seeds=candidates,
        probe_rule_idx=1,
    )
    assert len(selected) == 2
    assert set(selected).issubset(set(candidates))


def test_write_seed_selection_manifest(tmp_path: Path) -> None:
    write_seed_selection_manifest(
        out_root=tmp_path,
        mode="stratified",
        selected_seeds=[101, 202, 303],
        candidate_seeds=[100, 101, 102, 103],
        probe_rule_idx=1,
    )
    p = tmp_path / "doe_seed_selection.json"
    assert p.exists()
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["seed_mode"] == "stratified"
    assert payload["selected_seeds"] == [101, 202, 303]
    assert payload["probe_rule_idx"] == 1


def test_write_run_manifest(tmp_path: Path) -> None:
    plan = build_run_plan(
        out_root=tmp_path,
        design_points=[
            {
                "election_cost_rate": 0.01,
                "reward_rate_personal": 0.03,
                "break_even_distance_common": 0.2,
                "election_impact_on_mutation": 1.0,
                "mu": 0.3,
                "participation_alpha": 0.05,
                "participation_beta": 1.0,
                "participation_init_q": 0.0,
                "altruism_static": 0.5,
            }
        ],
        seeds=[101],
        primary_rule_idx=1,
        include_robustness=False,
    )
    write_run_manifest(out_root=tmp_path, plan=plan)
    p = tmp_path / "doe_run_manifest.csv"
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert "params_hash" in text
    assert "approval" in text


def test_rule_label_mapping_includes_schulze_before_random() -> None:
    assert rule_label(4) == "schulze"
    assert rule_label(5) == "random"
