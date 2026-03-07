from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import pytest

from src.analysis.doe_review_bundle import build_doe_review_bundle


def _make_fake_run_dir(base: Path, *, design_id: int, seed: int, rule_name: str = "approval") -> Path:
    run_dir = base / f"design_{design_id:04d}" / f"rule_{rule_name}" / f"seed_{seed:05d}" / "run_0"
    run_dir.mkdir(parents=True, exist_ok=True)
    static = {
        "personality_group_info": {
            "personality_groups": [[0, 1, 2], [1, 2, 0], [2, 0, 1]],
            "global_distribution": [0.45, 0.35, 0.20],
            "areas": {
                "0": {"num_agents": 60, "personality_group_distribution": [0.5, 0.3, 0.2]},
                "1": {"num_agents": 40, "personality_group_distribution": [0.4, 0.4, 0.2]},
            },
        }
    }
    (run_dir / "static.json").write_text(json.dumps(static), encoding="utf-8")
    return run_dir


def test_build_doe_review_bundle_writes_bundle_artifacts(tmp_path: Path) -> None:
    doe_root = tmp_path / "doe_20990101_000000"
    doe_root.mkdir(parents=True, exist_ok=True)

    design_scores = pd.DataFrame(
        [
            {"design_id": 0, "score_total": 0.92, "pass_rate": 1.00, "quality_mean": 0.77, "seed_robustness": 0.80},
            {"design_id": 1, "score_total": 0.84, "pass_rate": 0.95, "quality_mean": 0.68, "seed_robustness": 0.72},
            {"design_id": 2, "score_total": 0.51, "pass_rate": 0.65, "quality_mean": 0.42, "seed_robustness": 0.35},
            {"design_id": 3, "score_total": 0.44, "pass_rate": 0.55, "quality_mean": 0.37, "seed_robustness": 0.30},
            {"design_id": 4, "score_total": 0.18, "pass_rate": 0.20, "quality_mean": 0.14, "seed_robustness": 0.11},
            {"design_id": 5, "score_total": 0.07, "pass_rate": 0.00, "quality_mean": 0.05, "seed_robustness": 0.03},
        ]
    )
    design_scores.to_csv(doe_root / "doe_design_scores.csv", index=False)

    points = pd.DataFrame(
        [
            {"design_id": 0, "knob_a": 0.10, "knob_b": 10.0},
            {"design_id": 1, "knob_a": 0.20, "knob_b": 12.0},
            {"design_id": 2, "knob_a": 0.35, "knob_b": 14.0},
            {"design_id": 3, "knob_a": 0.55, "knob_b": 16.0},
            {"design_id": 4, "knob_a": 0.75, "knob_b": 18.0},
            {"design_id": 5, "knob_a": 0.90, "knob_b": 20.0},
        ]
    )
    points.to_csv(doe_root / "doe_design_points.csv", index=False)

    runs_rows = []
    for did in range(6):
        for seed in (101, 102):
            run_dir = _make_fake_run_dir(doe_root, design_id=did, seed=seed)
            runs_rows.append(
                {
                    "design_id": did,
                    "rule_name": "approval",
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "mean_turnout": 55.0 - did,
                    "mean_dist": 0.25 + 0.01 * did,
                    "winner_entropy_norm": 0.4 + 0.02 * did,
                    "competitive_step_share": 0.3 - 0.01 * did,
                    "roll20_group_turnout_range_max": 0.2 + 0.02 * did,
                    "puzzle_dominance_share_conflict": 0.6 - 0.05 * did,
                    "power_recovery_share_conflict": 0.2 + 0.03 * did,
                    "gate_puzzle_anti_monopoly": True,
                    "passes_hard_gates": did < 5,
                }
            )
    pd.DataFrame(runs_rows).to_csv(doe_root / "doe_run_features.csv", index=False)

    (doe_root / "doe_spec.json").write_text(
        json.dumps({"ranges": {"knob_a": [0.0, 1.0], "knob_b": [8.0, 22.0]}}, indent=2),
        encoding="utf-8",
    )

    artifacts = build_doe_review_bundle(
        doe_root=doe_root,
        per_bucket=2,
        rule_name="approval",
        render_summaries=False,
    )

    assert artifacts.bundle_root.exists()
    assert artifacts.queue_csv.exists()
    assert artifacts.bucket_manifest_csv.exists()
    assert artifacts.knob_ranges_csv.exists()
    assert artifacts.knob_correlations_csv.exists()
    assert artifacts.knob_effects_csv.exists()
    assert artifacts.analysis_summary_pdf.exists()

    queue = pd.read_csv(artifacts.queue_csv)
    assert {"bucket", "design_id", "representative_run_dir", "bundle_packet_dir", "bundle_overview_pdf"}.issubset(set(queue.columns))
    assert len(queue) >= 4
    assert {"top", "mid", "bottom"}.issubset(set(queue["bucket"].astype(str).unique()))

    for bucket in ("top", "mid", "bottom"):
        assert (artifacts.bundle_root / bucket).exists()

    knob_ranges = pd.read_csv(artifacts.knob_ranges_csv)
    assert {"knob", "range_min", "range_max"}.issubset(set(knob_ranges.columns))
    assert set(knob_ranges["knob"].tolist()) == {"knob_a", "knob_b"}


def test_build_doe_review_bundle_falls_back_when_design_scores_are_empty(tmp_path: Path) -> None:
    doe_root = tmp_path / "doe_20990101_000001"
    doe_root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        columns=[
            "design_id",
            "n_primary_runs",
            "n_primary_pass",
            "pass_rate",
            "quality_mean",
            "quality_std",
            "seed_robustness",
            "score_total",
        ]
    ).to_csv(doe_root / "doe_design_scores.csv", index=False)

    points = pd.DataFrame(
        [
            {"design_id": 0, "knob_a": 0.1, "knob_b": 1.0},
            {"design_id": 1, "knob_a": 0.5, "knob_b": 2.0},
            {"design_id": 2, "knob_a": 0.9, "knob_b": 3.0},
        ]
    )
    points.to_csv(doe_root / "doe_design_points.csv", index=False)

    runs_rows = []
    for did in (0, 1, 2):
        run_dir = _make_fake_run_dir(doe_root, design_id=did, seed=200 + did)
        runs_rows.append(
            {
                "design_id": did,
                "rule_name": "approval",
                "seed": 200 + did,
                "run_dir": str(run_dir),
                "mean_turnout": 50.0 - did,
                "winner_entropy_norm": 0.3 + 0.1 * did,
                "competitive_step_share": 0.2 + 0.05 * did,
                "roll20_group_turnout_range_max": 0.4 - 0.05 * did,
                "puzzle_dominance_share_conflict": 0.7 - 0.1 * did,
                "power_recovery_share_conflict": 0.1 + 0.1 * did,
                "gate_puzzle_anti_monopoly": True,
                "passes_hard_gates": did != 2,
            }
        )
    pd.DataFrame(runs_rows).to_csv(doe_root / "doe_run_features.csv", index=False)
    (doe_root / "doe_spec.json").write_text(
        json.dumps({"ranges": {"knob_a": [0.0, 1.0], "knob_b": [0.0, 4.0]}}, indent=2),
        encoding="utf-8",
    )

    artifacts = build_doe_review_bundle(
        doe_root=doe_root,
        per_bucket=1,
        rule_name="approval",
        render_summaries=False,
        allow_fallback_scores=True,
    )

    queue = pd.read_csv(artifacts.queue_csv)
    assert len(queue) == 3
    assert {"top", "mid", "bottom"} == set(queue["bucket"].astype(str).tolist())


def test_build_doe_review_bundle_fails_fast_when_design_scores_are_empty_and_fallback_disabled(tmp_path: Path) -> None:
    doe_root = tmp_path / "doe_20990101_000002"
    doe_root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(columns=["design_id", "score_total"]).to_csv(doe_root / "doe_design_scores.csv", index=False)
    pd.DataFrame([{"design_id": 0, "knob_a": 0.1}]).to_csv(doe_root / "doe_design_points.csv", index=False)
    run_dir = _make_fake_run_dir(doe_root, design_id=0, seed=300)
    pd.DataFrame(
        [
            {
                "design_id": 0,
                "rule_name": "approval",
                "seed": 300,
                "run_dir": str(run_dir),
                "passes_hard_gates": True,
            }
        ]
    ).to_csv(doe_root / "doe_run_features.csv", index=False)

    with pytest.raises(RuntimeError, match="doe_design_scores.csv is empty"):
        build_doe_review_bundle(
            doe_root=doe_root,
            per_bucket=1,
            rule_name="approval",
            render_summaries=False,
            allow_fallback_scores=False,
        )
