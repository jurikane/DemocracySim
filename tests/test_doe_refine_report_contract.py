from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import json

import pandas as pd

from src.analysis.doe_refine import build_refine_report


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


def test_build_refine_report_outputs_files(tmp_path: Path) -> None:
    root = tmp_path / "doe_mock"
    root.mkdir(parents=True, exist_ok=True)
    _write_mock_doe_tables(root)

    out = build_refine_report(root, elite_fraction=0.5, bad_fraction=0.5, shrink_quantile=0.8)
    assert out["knob_importance_csv"].exists()
    assert out["suggested_ranges_json"].exists()

    imp = pd.read_csv(out["knob_importance_csv"])
    assert {"knob", "corr_score", "elite_bad_std_effect"}.issubset(set(imp.columns))
    sug = json.loads(out["suggested_ranges_json"].read_text(encoding="utf-8"))
    assert "suggested_ranges" in sug
    assert "mu" in sug["suggested_ranges"]


def test_doe_refine_report_cli(tmp_path: Path) -> None:
    root = tmp_path / "doe_mock_cli"
    root.mkdir(parents=True, exist_ok=True)
    _write_mock_doe_tables(root)
    cmd = [
        sys.executable,
        "-m",
        "scripts.doe_refine_report",
        "--doe-root",
        str(root),
        "--elite-fraction",
        "0.5",
        "--bad-fraction",
        "0.5",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert (root / "doe_knob_importance.csv").exists()
    assert (root / "doe_suggested_ranges.json").exists()
