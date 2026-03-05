from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import json

import pandas as pd

from src.analysis.doe_refine import build_inference_report


def _write_mock_doe_tables(root: Path) -> None:
    pts = pd.DataFrame(
        [
            {"design_id": 0, "mu": 0.2, "participation_alpha": 0.02, "participation_beta": 1.0},
            {"design_id": 1, "mu": 0.4, "participation_alpha": 0.05, "participation_beta": 2.0},
            {"design_id": 2, "mu": 0.6, "participation_alpha": 0.08, "participation_beta": 4.0},
            {"design_id": 3, "mu": 0.8, "participation_alpha": 0.12, "participation_beta": 8.0},
        ]
    )
    scores = pd.DataFrame(
        [
            {"design_id": 0, "score_total": 0.10, "pass_rate": 0.33},
            {"design_id": 1, "score_total": 0.40, "pass_rate": 1.00},
            {"design_id": 2, "score_total": 0.55, "pass_rate": 1.00},
            {"design_id": 3, "score_total": 0.05, "pass_rate": 0.00},
        ]
    )
    pts.to_csv(root / "doe_design_points.csv", index=False)
    scores.to_csv(root / "doe_design_scores.csv", index=False)


def test_build_inference_report_outputs_files(tmp_path: Path) -> None:
    root = tmp_path / "doe_mock"
    root.mkdir(parents=True, exist_ok=True)
    _write_mock_doe_tables(root)

    out = build_inference_report(root, bootstrap_reps=50, random_seed=13)
    assert out["inference_spec_json"].exists()
    assert out["seed_fixed_effects_csv"].exists()
    assert out["nonlinear_importance_csv"].exists()
    assert out["interaction_maps_csv"].exists()
    assert out["bootstrap_design_ci_csv"].exists()
    assert out["pareto_designs_csv"].exists()
    assert (root / "doe_knob_importance.csv").exists() is False
    assert (root / "doe_suggested_ranges.json").exists() is False
    assert (root / "doe_inference_summary.json").exists() is False

    spec = json.loads(out["inference_spec_json"].read_text(encoding="utf-8"))
    assert "methods" in spec
    se = pd.read_csv(out["seed_fixed_effects_csv"])
    assert {"target", "knob", "coef", "ci_low", "ci_high"}.issubset(set(se.columns))
    nl = pd.read_csv(out["nonlinear_importance_csv"])
    assert {"target", "knob", "eta_squared"}.issubset(set(nl.columns))
    inter = pd.read_csv(out["interaction_maps_csv"])
    assert {"target", "knob_x", "knob_y", "x_bin", "y_bin", "cell_mean"}.issubset(set(inter.columns))
    boot = pd.read_csv(out["bootstrap_design_ci_csv"])
    assert {"design_id", "score_proxy_mean", "ci_low", "ci_high"}.issubset(set(boot.columns))
    pareto = pd.read_csv(out["pareto_designs_csv"])
    assert {"design_id", "is_pareto"}.issubset(set(pareto.columns))
    assert "seed_fixed_effects" in spec["methods"]


def test_doe_inference_cli(tmp_path: Path) -> None:
    root = tmp_path / "doe_mock_cli"
    root.mkdir(parents=True, exist_ok=True)
    _write_mock_doe_tables(root)
    cmd = [
        sys.executable,
        "-m",
        "tools.research.doe_inference",
        "--doe-root",
        str(root),
        "--bootstrap-reps",
        "50",
        "--random-seed",
        "13",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert (root / "doe_knob_importance.csv").exists() is False
    assert (root / "doe_suggested_ranges.json").exists() is False
    assert (root / "doe_inference_spec.json").exists()
    assert (root / "doe_seed_fixed_effects.csv").exists()
    assert (root / "doe_nonlinear_importance.csv").exists()
    assert (root / "doe_interaction_maps.csv").exists()
    assert (root / "doe_bootstrap_design_ci.csv").exists()
    assert (root / "doe_pareto_designs.csv").exists()
    assert (root / "doe_inference_summary.json").exists() is False
