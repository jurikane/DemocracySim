from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import json

import pandas as pd
import pytest

from src.analysis.doe_scoring import (
    _band_pref01,
    _upper_bound_pref01,
    analyze_doe_root,
    apply_hard_gates,
    compute_run_features_from_tables,
    load_selection_objective,
    score_designs,
)


def test_compute_run_features_from_tables_contract() -> None:
    area = pd.DataFrame(
        {
            "step": [1, 2, 3, 4, 5, 6, 7, 8],
            "participants": [0, 0, 1, 2, 2, 0, 1, 1],
            "turnout": [0.0, 0.0, 10.0, 20.0, 20.0, 0.0, 10.0, 10.0],
            "gini_index": [10.0, 11.0, 14.0, 16.0, 17.0, 18.0, 17.0, 16.0],
            "dist_to_reality": [0.4, 0.4, 0.3, 0.2, 0.2, 0.3, 0.25, 0.2],
            "winning_option_id": [1, 1, 1, 2, 2, 2, 1, 1],
        }
    )
    agents = pd.DataFrame(
        {
            "step": [1, 1, 2, 2, 3, 3, 4, 4],
            "personality_group_idx": [0, 1, 0, 1, 0, 1, 0, 1],
            "participating": [1, 0, 1, 0, 1, 0, 1, 0],
            "election_delta_rel": [0.20, -0.10, 0.30, -0.10, 0.15, -0.05, 0.10, -0.05],
        }
    )
    f = compute_run_features_from_tables(area, agents, burn_in_steps=2)
    assert f["max_all_abstain_stretch"] == 2
    assert f["winner_changes_post_burnin"] == 2
    assert f["roll3_group_turnout_range_max"] > 0.0
    assert f["roll20_group_turnout_range_max"] == 0.0
    assert f["roll10_dist_std_mean"] == 0.0
    assert f["roll10_winner_change_rate"] == 0.0
    assert f["roll10_turnout_slope_abs_mean"] == 0.0
    assert f["roll10_group_sync_index"] == 0.0
    assert -1.0 <= f["lag1_group_signal_turnout_response_corr"] <= 1.0
    assert 0.0 <= f["winner_entropy_norm"] <= 1.0
    assert f["dist_nonzero_share"] > 0.0
    assert f["turnout_std"] > 0.0
    assert 0.0 <= f["turnout_start_window_mean"] <= 100.0
    assert 0.0 <= f["turnout_end_window_mean"] <= 100.0
    assert f["turnout_drop_start_end"] >= -100.0
    assert 0.0 <= f["turnout_outside_20_80_share"] <= 1.0
    assert f["turnout_decline_slope_norm"] >= 0.0
    assert f["group_participation_std"] > 0.0
    assert f["participant_abstainer_delta_rel_gap_abs"] > 0.0
    assert f["participant_share_max_abs_drift_20"] >= 0.0
    assert f["participant_share_mean_abs_drift_20_w"] >= 0.0
    assert f["participant_share_turnover_rate_w"] >= 0.0


def test_turnout_shape_score_helpers_are_not_limited_to_unit_scale() -> None:
    s = pd.Series([10.0, 50.0, 90.0])
    band = _band_pref01(s, low=20.0, high=80.0)
    assert float(band.iloc[1]) == 1.0
    assert 0.0 < float(band.iloc[0]) < 1.0
    assert 0.0 < float(band.iloc[2]) < 1.0

    ub = _upper_bound_pref01(pd.Series([5.0, 20.0, 60.0, 80.0]), good_max=20.0, zero_at=70.0)
    assert float(ub.iloc[0]) == 1.0
    assert float(ub.iloc[1]) == 1.0
    assert 0.0 < float(ub.iloc[2]) < 1.0
    assert float(ub.iloc[3]) == 0.0


def test_apply_hard_gates_contract() -> None:
    runs = pd.DataFrame(
        [
            {
                "design_id": 0,
                "rule_name": "approval",
                "seed": 101,
                "max_all_abstain_stretch": 0,
                "winner_changes_post_burnin": 5,
                "winner_change_rate_post_burnin": 0.05,
                "turnout_std": 2.0,
                "gini_std": 4.0,
                "dist_std": 0.1,
                "group_turnout_range_mean": 0.08,
                "roll3_group_turnout_range_max": 0.35,
                "roll20_group_turnout_range_max": 0.60,
                "winner_entropy_norm": 0.6,
                "dist_nonzero_share": 0.3,
                "competitive_step_share": 0.2,
                "mean_turnout": 60.0,
            },
            {
                "design_id": 1,
                "rule_name": "approval",
                "seed": 101,
                "max_all_abstain_stretch": 40,
                "winner_changes_post_burnin": 0,
                "winner_change_rate_post_burnin": 0.0,
                "turnout_std": 0.0,
                "gini_std": 0.0,
                "dist_std": 0.0,
                "group_turnout_range_mean": 0.01,
                "roll3_group_turnout_range_max": 0.05,
                "roll20_group_turnout_range_max": 0.01,
                "winner_entropy_norm": 0.0,
                "dist_nonzero_share": 0.0,
                "competitive_step_share": 0.0,
                "mean_turnout": 95.0,
            },
            {
                "design_id": 2,
                "rule_name": "approval",
                "seed": 101,
                "max_all_abstain_stretch": 0,
                "winner_changes_post_burnin": 500,
                "winner_change_rate_post_burnin": 0.90,
                "turnout_std": 1.0,
                "gini_std": 2.0,
                "dist_std": 0.2,
                "group_turnout_range_mean": 0.08,
                "roll3_group_turnout_range_max": 0.35,
                "roll20_group_turnout_range_max": 0.60,
                "winner_entropy_norm": 0.8,
                "dist_nonzero_share": 0.4,
                "competitive_step_share": 0.4,
                "mean_turnout": 60.0,
            },
        ]
    )
    out = apply_hard_gates(
        runs,
        max_all_abstain_stretch=20,
        min_winner_changes_post_burnin=3,
        max_winner_changes_post_burnin=100,
        min_group_turnout_range_mean=0.03,
        min_roll3_group_turnout_range_max=0.3,
        min_roll20_group_turnout_range_max=0.3,
        min_turnout_std=0.5,
        min_gini_std=1.0,
        min_dist_std=0.02,
        min_winner_entropy_norm=0.2,
        min_dist_nonzero_share=0.05,
        min_competitive_step_share=0.05,
        min_mean_turnout=20.0,
        max_mean_turnout=90.0,
    )
    assert bool(out.loc[out["design_id"] == 0, "passes_hard_gates"].iloc[0]) is True
    assert bool(out.loc[out["design_id"] == 1, "passes_hard_gates"].iloc[0]) is False
    assert bool(out.loc[out["design_id"] == 2, "passes_hard_gates"].iloc[0]) is False


def test_apply_hard_gates_lockin_is_hard_blocker() -> None:
    runs = pd.DataFrame(
        [
            {
                "design_id": 0,
                "rule_name": "approval",
                "seed": 101,
                "max_all_abstain_stretch": 0,
                "winner_changes_post_burnin": 1,  # below lock-in threshold
                "winner_change_rate_post_burnin": 0.05,
                "turnout_std": 2.0,
                "gini_std": 4.0,
                "dist_std": 0.1,
                "group_turnout_range_mean": 0.01,  # below group-divergence threshold
                "roll3_group_turnout_range_max": 0.35,
                "roll20_group_turnout_range_max": 0.60,
                "winner_entropy_norm": 0.60,
                "dist_nonzero_share": 0.30,
                "competitive_step_share": 0.20,
                "mean_turnout": 60.0,
            },
        ]
    )
    out = apply_hard_gates(
        runs,
        max_all_abstain_stretch=10,
        min_winner_changes_post_burnin=2,
        max_winner_changes_post_burnin=120,
        min_group_turnout_range_mean=0.03,
        min_roll3_group_turnout_range_max=0.3,
        min_roll20_group_turnout_range_max=0.3,
        min_turnout_std=0.5,
        min_gini_std=1.0,
        min_dist_std=0.02,
        min_winner_entropy_norm=0.2,
        min_dist_nonzero_share=0.05,
        min_competitive_step_share=0.05,
        min_mean_turnout=20.0,
        max_mean_turnout=90.0,
    )
    assert bool(out.loc[0, "gate_no_lockin"]) is False
    assert bool(out.loc[0, "passes_hard_gates"]) is False


def test_apply_hard_gates_requires_roll3_and_roll20() -> None:
    runs = pd.DataFrame(
        [
            {
                "design_id": 0,
                "rule_name": "approval",
                "seed": 101,
                "max_all_abstain_stretch": 0,
                "winner_changes_post_burnin": 6,
                "winner_change_rate_post_burnin": 0.06,
                "turnout_std": 2.0,
                "gini_std": 4.0,
                "dist_std": 0.1,
                "group_turnout_range_mean": 0.08,
                "roll3_group_turnout_range_max": 0.35,  # passes roll3
                "roll20_group_turnout_range_max": 0.08,  # fails roll20
                "winner_entropy_norm": 0.6,
                "dist_nonzero_share": 0.3,
                "competitive_step_share": 0.2,
                "mean_turnout": 60.0,
            },
        ]
    )
    out = apply_hard_gates(
        runs,
        max_all_abstain_stretch=10,
        min_winner_changes_post_burnin=3,
        max_winner_changes_post_burnin=120,
        min_group_turnout_range_mean=0.03,
        min_roll3_group_turnout_range_max=0.3,
        min_roll20_group_turnout_range_max=0.1,
        min_turnout_std=0.5,
        min_gini_std=1.0,
        min_dist_std=0.02,
        min_winner_entropy_norm=0.2,
        min_dist_nonzero_share=0.05,
        min_competitive_step_share=0.05,
        min_mean_turnout=20.0,
        max_mean_turnout=90.0,
    )
    assert bool(out.loc[0, "gate_roll3_divergence"]) is True
    assert bool(out.loc[0, "gate_roll20_divergence"]) is False
    assert bool(out.loc[0, "passes_hard_gates"]) is False


def test_apply_hard_gates_fails_fast_on_missing_required_columns() -> None:
    runs = pd.DataFrame(
        [
            {
                "design_id": 0,
                "rule_name": "approval",
                "seed": 101,
                "max_all_abstain_stretch": 0,
                "winner_changes_post_burnin": 5,
                "turnout_std": 2.0,
                "gini_std": 4.0,
                "dist_std": 0.1,
                "group_turnout_range_mean": 0.08,
                "roll3_group_turnout_range_max": 0.35,
                "roll20_group_turnout_range_max": 0.60,
                "winner_entropy_norm": 0.6,
                "dist_nonzero_share": 0.3,
                "competitive_step_share": 0.2,
                "mean_turnout": 60.0,
            },
        ]
    )
    with pytest.raises(ValueError, match="missing required columns"):
        apply_hard_gates(runs)


def test_score_designs_fails_fast_on_missing_required_columns() -> None:
    runs = pd.DataFrame(
        [
            {
                "design_id": 0,
                "rule_name": "approval",
                "seed": 101,
                "passes_hard_gates": True,
                "turnout_std": 1.0,
                "gini_std": 1.0,
                "dist_std": 0.1,
                "group_participation_std": 0.1,
                "group_turnout_range_mean": 0.1,
                "roll3_group_turnout_range_mean": 0.1,
                "roll3_group_turnout_range_max": 0.2,
                "roll20_group_turnout_range_mean": 0.1,
                "roll20_group_turnout_range_max": 0.2,
                "group_turnout_residual_abs_mean": 0.1,
                "participant_abstainer_delta_rel_gap_abs": 0.1,
                "group_participant_abstainer_delta_rel_gap_abs": 0.1,
                "winner_entropy_norm": 0.4,
                "dist_nonzero_share": 0.2,
                "competitive_step_share": 0.2,
                "winner_change_rate_post_burnin": 0.05,
                "mean_turnout": 50.0,
                "turnout_start_window_mean": 52.0,
                # turnout_end_window_mean intentionally missing
                "turnout_drop_start_end": 3.0,
                "turnout_decline_slope_norm": 0.05,
                "turnout_outside_20_80_share": 0.04,
            },
        ]
    )
    with pytest.raises(ValueError, match="missing required columns"):
        score_designs(runs)


def test_score_designs_optional_metric_diagnostics_are_explicit() -> None:
    runs = pd.DataFrame(
        [
            {
                "design_id": 0,
                "rule_name": "approval",
                "seed": 101,
                "passes_hard_gates": True,
                "turnout_std": 1.0,
                "gini_std": 1.0,
                "dist_std": 0.1,
                "group_participation_std": 0.1,
                "group_turnout_range_mean": 0.1,
                "roll3_group_turnout_range_mean": 0.1,
                "roll3_group_turnout_range_max": 0.2,
                "roll20_group_turnout_range_mean": 0.1,
                "roll20_group_turnout_range_max": 0.2,
                "group_turnout_residual_abs_mean": 0.1,
                "participant_abstainer_delta_rel_gap_abs": 0.1,
                "group_participant_abstainer_delta_rel_gap_abs": 0.1,
                "winner_entropy_norm": 0.4,
                "dist_nonzero_share": 0.2,
                "competitive_step_share": 0.2,
                "winner_change_rate_post_burnin": 0.05,
                "mean_turnout": 50.0,
                "turnout_start_window_mean": 52.0,
                "turnout_end_window_mean": 49.0,
                "turnout_drop_start_end": 3.0,
                "turnout_decline_slope_norm": 0.05,
                "turnout_outside_20_80_share": 0.04,
            },
            {
                "design_id": 0,
                "rule_name": "approval",
                "seed": 102,
                "passes_hard_gates": True,
                "turnout_std": 1.1,
                "gini_std": 1.1,
                "dist_std": 0.11,
                "group_participation_std": 0.11,
                "group_turnout_range_mean": 0.11,
                "roll3_group_turnout_range_mean": 0.11,
                "roll3_group_turnout_range_max": 0.21,
                "roll20_group_turnout_range_mean": 0.11,
                "roll20_group_turnout_range_max": 0.21,
                "group_turnout_residual_abs_mean": 0.11,
                "participant_abstainer_delta_rel_gap_abs": 0.11,
                "group_participant_abstainer_delta_rel_gap_abs": 0.11,
                "winner_entropy_norm": 0.41,
                "dist_nonzero_share": 0.21,
                "competitive_step_share": 0.21,
                "winner_change_rate_post_burnin": 0.06,
                "mean_turnout": 51.0,
                "turnout_start_window_mean": 53.0,
                "turnout_end_window_mean": 50.0,
                "turnout_drop_start_end": 3.0,
                "turnout_decline_slope_norm": 0.05,
                "turnout_outside_20_80_share": 0.04,
            },
        ]
    )

    _, meta = score_designs(runs, return_meta=True)
    diag = meta["optional_metric_diagnostics"]
    assert "participation_q_delta_mean_abs" in diag["missing_optional_columns"]
    assert int(diag["optional_na_counts"]["participation_q_delta_mean_abs"]) == 2


def _write_run_tables(run_dir: Path, *, participants_scale: float, dist_scale: float, delta_rel_amp: float = 0.2) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    steps = list(range(1, 11))
    participants = [int(participants_scale * x) for x in [5, 6, 7, 8, 7, 6, 5, 4, 5, 6]]
    area = pd.DataFrame(
        {
            "step": steps,
            "participants": participants,
            "turnout": [float(p) for p in participants],
            "gini_index": [20.0, 21.0, 22.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0],
            "dist_to_reality": [dist_scale * x for x in [0.3, 0.25, 0.2, 0.15, 0.2, 0.3, 0.35, 0.3, 0.25, 0.2]],
            "winning_option_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        }
    )
    area.to_parquet(run_dir / "area_steps.parquet")
    agents = pd.DataFrame(
        {
            "step": [1, 1, 2, 2, 3, 3, 4, 4],
            "personality_group_idx": [0, 1, 0, 1, 0, 1, 0, 1],
            "participating": [1, 0, 1, 0, 1, 0, 1, 0],
            "election_delta_rel": [delta_rel_amp, -delta_rel_amp, delta_rel_amp, -delta_rel_amp, delta_rel_amp, -delta_rel_amp, delta_rel_amp, -delta_rel_amp],
            "eligible_for_election": [True, True, True, True, True, True, True, True],
        }
    )
    agents.to_parquet(run_dir / "agents.parquet")


def test_analyze_doe_root_and_score_designs(tmp_path: Path) -> None:
    root = tmp_path / "doe_x"
    # design 0: stronger dynamics + stronger approval/utilitarian separation
    _write_run_tables(root / "design_0000" / "rule_approval" / "seed_00101" / "run_0", participants_scale=1.0, dist_scale=1.0, delta_rel_amp=0.4)
    _write_run_tables(root / "design_0000" / "rule_approval" / "seed_00102" / "run_0", participants_scale=1.0, dist_scale=1.0, delta_rel_amp=0.4)
    _write_run_tables(root / "design_0000" / "rule_utilitarian" / "seed_00101" / "run_0", participants_scale=0.7, dist_scale=0.4, delta_rel_amp=0.3)
    _write_run_tables(root / "design_0000" / "rule_utilitarian" / "seed_00102" / "run_0", participants_scale=0.7, dist_scale=0.4, delta_rel_amp=0.3)
    # design 1: weaker dynamics
    _write_run_tables(root / "design_0001" / "rule_approval" / "seed_00101" / "run_0", participants_scale=0.3, dist_scale=0.2)
    _write_run_tables(root / "design_0001" / "rule_approval" / "seed_00102" / "run_0", participants_scale=0.3, dist_scale=0.2)
    _write_run_tables(root / "design_0001" / "rule_utilitarian" / "seed_00101" / "run_0", participants_scale=0.3, dist_scale=0.2)
    _write_run_tables(root / "design_0001" / "rule_utilitarian" / "seed_00102" / "run_0", participants_scale=0.3, dist_scale=0.2)

    out = analyze_doe_root(
        root,
        burn_in_steps=2,
        thresholds={
            "max_all_abstain_stretch": 999.0,
            "min_winner_changes_post_burnin": 0.0,
            "max_winner_changes_post_burnin": 999.0,
            "min_group_turnout_range_mean": 0.0,
            "min_roll3_group_turnout_range_max": 0.0,
            "min_roll20_group_turnout_range_max": 0.0,
            "min_turnout_std": 0.0,
            "min_gini_std": 0.0,
            "min_dist_std": 0.0,
        },
    )
    assert out["run_features_csv"].exists()
    assert out["design_scores_csv"].exists()
    assert out["selection_spec_json"].exists()
    assert out["top_designs_json"].exists()

    rf = pd.read_csv(out["run_features_csv"])
    gated = apply_hard_gates(rf)
    scores = score_designs(gated)
    assert len(scores) >= 2
    best_design = int(scores.sort_values("score_total", ascending=False).iloc[0]["design_id"])
    assert best_design == 0


def test_score_doe_cli_writes_outputs(tmp_path: Path) -> None:
    root = tmp_path / "doe_cli"
    _write_run_tables(root / "design_0000" / "rule_approval" / "seed_00101" / "run_0", participants_scale=1.0, dist_scale=1.0)
    _write_run_tables(root / "design_0000" / "rule_utilitarian" / "seed_00101" / "run_0", participants_scale=0.7, dist_scale=0.4)
    cmd = [
        sys.executable,
        "-m",
        "scripts.score_doe",
        "--doe-root",
        str(root),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert (root / "doe_run_features.csv").exists()
    assert (root / "doe_design_scores.csv").exists()
    assert (root / "doe_selection_spec.json").exists()
    assert (root / "doe_top_designs.json").exists()
    assert (root / "doe_scoring_spec.json").exists() is False
    spec = json.loads((root / "doe_selection_spec.json").read_text(encoding="utf-8"))
    assert int(spec["burn_in_steps"]) == 0
    assert "effective_weights" in spec


def test_load_selection_objective_contract(tmp_path: Path) -> None:
    p = tmp_path / "objective.json"
    p.write_text(
        json.dumps(
            {
                "version": "v1",
                "thresholds": {"max_all_abstain_stretch": 7.0},
                "weights": {"quality_mean": 0.5, "discriminability": 0.3, "seed_robustness": 0.2},
                "stage_weights": {"viability": 0.7, "quality_bundle": 0.3},
                "strict_completeness": False,
            }
        ),
        encoding="utf-8",
    )
    obj = load_selection_objective(p)
    assert float(obj["thresholds"]["max_all_abstain_stretch"]) == 7.0
    assert float(obj["weights"]["quality_mean"]) == 0.5
    assert float(obj["stage_weights"]["viability"]) == 0.7
    assert bool(obj["strict_completeness"]) is False


def test_analyze_doe_root_reads_objective_contract(tmp_path: Path) -> None:
    root = tmp_path / "doe_obj"
    _write_run_tables(root / "design_0000" / "rule_approval" / "seed_00101" / "run_0", participants_scale=1.0, dist_scale=1.0)
    _write_run_tables(root / "design_0000" / "rule_utilitarian" / "seed_00101" / "run_0", participants_scale=0.7, dist_scale=0.4)
    objective = tmp_path / "objective.json"
    objective.write_text(
        json.dumps(
            {
                "version": "v1",
                "thresholds": {
                    "max_all_abstain_stretch": 999.0,
                    "min_winner_changes_post_burnin": 0.0,
                    "max_winner_changes_post_burnin": 999.0,
                    "min_group_turnout_range_mean": 0.0,
                    "min_roll3_group_turnout_range_max": 0.0,
                    "min_roll20_group_turnout_range_max": 0.0,
                    "min_turnout_std": 0.0,
                    "min_gini_std": 0.0,
                    "min_dist_std": 0.0,
                    "min_winner_entropy_norm": 0.0,
                    "min_dist_nonzero_share": 0.0,
                    "min_competitive_step_share": 0.0,
                    "min_mean_turnout": 0.0,
                    "max_mean_turnout": 100.0,
                },
                "weights": {"quality_mean": 0.45, "discriminability": 0.35, "seed_robustness": 0.2},
                "stage_weights": {"viability": 0.6, "quality_bundle": 0.4},
                "strict_completeness": False,
            }
        ),
        encoding="utf-8",
    )
    out = analyze_doe_root(root, objective_config_path=objective)
    spec = json.loads(out["selection_spec_json"].read_text(encoding="utf-8"))
    assert spec["objective_config_path"].endswith("objective.json")
    assert bool(spec["strict_completeness"]) is False
    assert float(spec["thresholds"]["max_all_abstain_stretch"]) == 999.0


def test_score_designs_reports_selection_score_only() -> None:
    runs = pd.DataFrame(
        [
            {
                "design_id": 0, "rule_name": "approval", "seed": 101, "passes_hard_gates": True,
                "turnout_std": 2.0, "gini_std": 2.0, "dist_std": 0.2,
                "group_participation_std": 0.2, "group_turnout_range_mean": 0.2,
                "roll3_group_turnout_range_mean": 0.2, "roll3_group_turnout_range_max": 0.3,
                "roll20_group_turnout_range_mean": 0.2, "roll20_group_turnout_range_max": 0.3,
                "group_turnout_residual_abs_mean": 0.2,
                "participant_abstainer_delta_rel_gap_abs": 0.2, "group_participant_abstainer_delta_rel_gap_abs": 0.2,
                "winner_entropy_norm": 0.5, "dist_nonzero_share": 0.3, "competitive_step_share": 0.3,
                "winner_changes_post_burnin": 8.0, "winner_change_rate_post_burnin": 0.08,
                "mean_turnout": 55.0, "turnout_start_window_mean": 58.0, "turnout_end_window_mean": 52.0,
                "turnout_drop_start_end": 6.0, "turnout_decline_slope_norm": 0.08, "turnout_outside_20_80_share": 0.10,
                "mean_gini": 20.0, "mean_dist": 0.2,
            },
            {
                "design_id": 0, "rule_name": "approval", "seed": 102, "passes_hard_gates": False,
                "turnout_std": 1.0, "gini_std": 1.0, "dist_std": 0.1,
                "group_participation_std": 0.1, "group_turnout_range_mean": 0.1,
                "roll3_group_turnout_range_mean": 0.1, "roll3_group_turnout_range_max": 0.2,
                "roll20_group_turnout_range_mean": 0.1, "roll20_group_turnout_range_max": 0.2,
                "group_turnout_residual_abs_mean": 0.1,
                "participant_abstainer_delta_rel_gap_abs": 0.1, "group_participant_abstainer_delta_rel_gap_abs": 0.1,
                "winner_entropy_norm": 0.3, "dist_nonzero_share": 0.2, "competitive_step_share": 0.2,
                "winner_changes_post_burnin": 5.0, "winner_change_rate_post_burnin": 0.05,
                "mean_turnout": 65.0, "turnout_start_window_mean": 67.0, "turnout_end_window_mean": 63.0,
                "turnout_drop_start_end": 4.0, "turnout_decline_slope_norm": 0.06, "turnout_outside_20_80_share": 0.08,
                "mean_gini": 21.0, "mean_dist": 0.25,
            },
            {
                "design_id": 1, "rule_name": "approval", "seed": 101, "passes_hard_gates": True,
                "turnout_std": 1.5, "gini_std": 1.5, "dist_std": 0.15,
                "group_participation_std": 0.15, "group_turnout_range_mean": 0.15,
                "roll3_group_turnout_range_mean": 0.15, "roll3_group_turnout_range_max": 0.25,
                "roll20_group_turnout_range_mean": 0.15, "roll20_group_turnout_range_max": 0.25,
                "group_turnout_residual_abs_mean": 0.15,
                "participant_abstainer_delta_rel_gap_abs": 0.15, "group_participant_abstainer_delta_rel_gap_abs": 0.15,
                "winner_entropy_norm": 0.4, "dist_nonzero_share": 0.25, "competitive_step_share": 0.25,
                "winner_changes_post_burnin": 6.0, "winner_change_rate_post_burnin": 0.06,
                "mean_turnout": 50.0, "turnout_start_window_mean": 52.0, "turnout_end_window_mean": 48.0,
                "turnout_drop_start_end": 4.0, "turnout_decline_slope_norm": 0.06, "turnout_outside_20_80_share": 0.06,
                "mean_gini": 19.0, "mean_dist": 0.18,
            },
            {
                "design_id": 1, "rule_name": "approval", "seed": 102, "passes_hard_gates": True,
                "turnout_std": 1.4, "gini_std": 1.4, "dist_std": 0.14,
                "group_participation_std": 0.14, "group_turnout_range_mean": 0.14,
                "roll3_group_turnout_range_mean": 0.14, "roll3_group_turnout_range_max": 0.24,
                "roll20_group_turnout_range_mean": 0.14, "roll20_group_turnout_range_max": 0.24,
                "group_turnout_residual_abs_mean": 0.14,
                "participant_abstainer_delta_rel_gap_abs": 0.14, "group_participant_abstainer_delta_rel_gap_abs": 0.14,
                "winner_entropy_norm": 0.35, "dist_nonzero_share": 0.24, "competitive_step_share": 0.24,
                "winner_changes_post_burnin": 5.5, "winner_change_rate_post_burnin": 0.055,
                "mean_turnout": 52.0, "turnout_start_window_mean": 54.0, "turnout_end_window_mean": 50.0,
                "turnout_drop_start_end": 4.0, "turnout_decline_slope_norm": 0.06, "turnout_outside_20_80_share": 0.06,
                "mean_gini": 19.5, "mean_dist": 0.19,
            },
        ]
    )
    out = score_designs(runs)
    assert "score_total" in out.columns
    assert "legacy_score_total" not in out.columns


def test_score_designs_required_primary_runs_filters_incomplete_designs() -> None:
    runs = pd.DataFrame(
        [
            # design 0 complete primary coverage (2 seeds)
            {
                "design_id": 0, "rule_name": "approval", "seed": 101, "passes_hard_gates": True,
                "turnout_std": 1.0, "gini_std": 1.0, "dist_std": 0.1,
                "group_participation_std": 0.1, "group_turnout_range_mean": 0.1,
                "roll3_group_turnout_range_mean": 0.1, "roll3_group_turnout_range_max": 0.2,
                "roll20_group_turnout_range_mean": 0.1, "roll20_group_turnout_range_max": 0.2,
                "group_turnout_residual_abs_mean": 0.1,
                "participant_abstainer_delta_rel_gap_abs": 0.1, "group_participant_abstainer_delta_rel_gap_abs": 0.1,
                "winner_entropy_norm": 0.4, "dist_nonzero_share": 0.2, "competitive_step_share": 0.2,
                "winner_changes_post_burnin": 5.0, "winner_change_rate_post_burnin": 0.05,
                "mean_turnout": 50.0, "turnout_start_window_mean": 52.0, "turnout_end_window_mean": 49.0,
                "turnout_drop_start_end": 3.0, "turnout_decline_slope_norm": 0.05, "turnout_outside_20_80_share": 0.04,
                "mean_gini": 20.0, "mean_dist": 0.2,
            },
            {
                "design_id": 0, "rule_name": "approval", "seed": 102, "passes_hard_gates": True,
                "turnout_std": 1.0, "gini_std": 1.0, "dist_std": 0.1,
                "group_participation_std": 0.1, "group_turnout_range_mean": 0.1,
                "roll3_group_turnout_range_mean": 0.1, "roll3_group_turnout_range_max": 0.2,
                "roll20_group_turnout_range_mean": 0.1, "roll20_group_turnout_range_max": 0.2,
                "group_turnout_residual_abs_mean": 0.1,
                "participant_abstainer_delta_rel_gap_abs": 0.1, "group_participant_abstainer_delta_rel_gap_abs": 0.1,
                "winner_entropy_norm": 0.4, "dist_nonzero_share": 0.2, "competitive_step_share": 0.2,
                "winner_changes_post_burnin": 5.0, "winner_change_rate_post_burnin": 0.05,
                "mean_turnout": 50.0, "turnout_start_window_mean": 52.0, "turnout_end_window_mean": 49.0,
                "turnout_drop_start_end": 3.0, "turnout_decline_slope_norm": 0.05, "turnout_outside_20_80_share": 0.04,
                "mean_gini": 20.0, "mean_dist": 0.2,
            },
            # design 1 incomplete primary coverage (1 seed)
            {
                "design_id": 1, "rule_name": "approval", "seed": 101, "passes_hard_gates": True,
                "turnout_std": 1.0, "gini_std": 1.0, "dist_std": 0.1,
                "group_participation_std": 0.1, "group_turnout_range_mean": 0.1,
                "roll3_group_turnout_range_mean": 0.1, "roll3_group_turnout_range_max": 0.2,
                "roll20_group_turnout_range_mean": 0.1, "roll20_group_turnout_range_max": 0.2,
                "group_turnout_residual_abs_mean": 0.1,
                "participant_abstainer_delta_rel_gap_abs": 0.1, "group_participant_abstainer_delta_rel_gap_abs": 0.1,
                "winner_entropy_norm": 0.4, "dist_nonzero_share": 0.2, "competitive_step_share": 0.2,
                "winner_changes_post_burnin": 5.0, "winner_change_rate_post_burnin": 0.05,
                "mean_turnout": 50.0, "turnout_start_window_mean": 52.0, "turnout_end_window_mean": 49.0,
                "turnout_drop_start_end": 3.0, "turnout_decline_slope_norm": 0.05, "turnout_outside_20_80_share": 0.04,
                "mean_gini": 20.0, "mean_dist": 0.2,
            },
        ]
    )
    out = score_designs(runs, required_primary_runs=2)
    assert set(out["design_id"].tolist()) == {0}


def test_analyze_doe_root_strict_completeness_filters_incomplete_designs(tmp_path: Path) -> None:
    root = tmp_path / "doe_incomplete"
    # design 0 complete for two seeds in both rules
    _write_run_tables(root / "design_0000" / "rule_approval" / "seed_00101" / "run_0", participants_scale=1.0, dist_scale=1.0)
    _write_run_tables(root / "design_0000" / "rule_approval" / "seed_00102" / "run_0", participants_scale=1.0, dist_scale=1.0)
    _write_run_tables(root / "design_0000" / "rule_utilitarian" / "seed_00101" / "run_0", participants_scale=0.7, dist_scale=0.4)
    _write_run_tables(root / "design_0000" / "rule_utilitarian" / "seed_00102" / "run_0", participants_scale=0.7, dist_scale=0.4)
    # design 1 incomplete: only one seed
    _write_run_tables(root / "design_0001" / "rule_approval" / "seed_00101" / "run_0", participants_scale=1.0, dist_scale=1.0)
    _write_run_tables(root / "design_0001" / "rule_utilitarian" / "seed_00101" / "run_0", participants_scale=0.7, dist_scale=0.4)

    (root / "doe_spec.json").write_text(
        json.dumps(
            {
                "seeds": [101, 102],
                "include_robustness": True,
                "robust_every": 1,
            }
        ),
        encoding="utf-8",
    )

    out = analyze_doe_root(
        root,
        burn_in_steps=0,
        thresholds={
            "max_all_abstain_stretch": 999.0,
            "min_winner_changes_post_burnin": 0.0,
            "max_winner_changes_post_burnin": 999.0,
            "min_group_turnout_range_mean": 0.0,
            "min_roll3_group_turnout_range_max": 0.0,
            "min_roll20_group_turnout_range_max": 0.0,
            "min_turnout_std": 0.0,
            "min_gini_std": 0.0,
            "min_dist_std": 0.0,
            "min_winner_entropy_norm": 0.0,
            "min_dist_nonzero_share": 0.0,
            "min_competitive_step_share": 0.0,
            "min_mean_turnout": 0.0,
            "max_mean_turnout": 100.0,
        },
    )
    ds = pd.read_csv(out["design_scores_csv"])
    assert set(ds["design_id"].tolist()) == {0}
