from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import json

import pandas as pd

from src.analysis.doe_scoring import (
    analyze_doe_root,
    apply_hard_gates,
    compute_run_features_from_tables,
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
    assert f["roll20_group_turnout_range_max"] == 0.0
    assert 0.0 <= f["winner_entropy_norm"] <= 1.0
    assert f["dist_nonzero_share"] > 0.0
    assert f["turnout_std"] > 0.0
    assert f["group_participation_std"] > 0.0
    assert f["participant_abstainer_delta_rel_gap_abs"] > 0.0


def test_apply_hard_gates_contract() -> None:
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
                "turnout_std": 0.0,
                "gini_std": 0.0,
                "dist_std": 0.0,
                "group_turnout_range_mean": 0.01,
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
                "turnout_std": 1.0,
                "gini_std": 2.0,
                "dist_std": 0.2,
                "group_turnout_range_mean": 0.08,
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
                "turnout_std": 2.0,
                "gini_std": 4.0,
                "dist_std": 0.1,
                "group_turnout_range_mean": 0.01,  # below group-divergence threshold
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
            "min_roll20_group_turnout_range_max": 0.0,
            "min_turnout_std": 0.0,
            "min_gini_std": 0.0,
            "min_dist_std": 0.0,
        },
    )
    assert out["run_features_csv"].exists()
    assert out["design_scores_csv"].exists()
    assert out["scoring_spec_json"].exists()
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
    assert (root / "doe_scoring_spec.json").exists()
    assert (root / "doe_top_designs.json").exists()
    spec = json.loads((root / "doe_scoring_spec.json").read_text(encoding="utf-8"))
    assert int(spec["burn_in_steps"]) == 0
    assert "effective_weights" in spec
