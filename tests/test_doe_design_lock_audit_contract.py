from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from scripts.repro import audit_doe_design_lock as audit_lock
from src.analysis.doe_scoring import score_designs


def _f(x: object) -> float:
    return float(x)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_doe_bundle_reproduces_locked_top_design() -> None:
    bundle_dir = Path("configs/thesis/doe_selection_bundle_v1")
    selection = json.loads((bundle_dir / "doe_selection_spec.json").read_text(encoding="utf-8"))
    provenance = json.loads(Path("configs/thesis/doe_selection_provenance_v1.json").read_text(encoding="utf-8"))
    run_features = pd.read_csv(bundle_dir / "doe_run_features.csv")

    thresholds = selection["thresholds"]

    scores = score_designs(
        run_features,
        primary_rule_name=str(selection["primary_rule_name"]),
        weights=dict(selection["weights"]),
        stage_weights=dict(selection["stage_weights"]),
        required_primary_runs=(None if selection.get("required_primary_runs") is None else int(selection["required_primary_runs"])),
        min_winner_entropy_norm=_f(thresholds["min_winner_entropy_norm"]),
        min_competitive_step_share=_f(thresholds["min_competitive_step_share"]),
        puzzle_dominance_share_score_low=_f(thresholds["puzzle_dominance_share_score_low"]),
        puzzle_dominance_share_score_high=_f(thresholds["puzzle_dominance_share_score_high"]),
        turnout_start_score_low=_f(thresholds["turnout_start_score_low"]),
        turnout_start_score_high=_f(thresholds["turnout_start_score_high"]),
        turnout_end_score_low=_f(thresholds["turnout_end_score_low"]),
        turnout_end_score_high=_f(thresholds["turnout_end_score_high"]),
        turnout_drop_score_good_max=_f(thresholds["turnout_drop_score_good_max"]),
        turnout_drop_score_zero_at=_f(thresholds["turnout_drop_score_zero_at"]),
        turnout_decline_score_good_max=_f(thresholds["turnout_decline_score_good_max"]),
        turnout_decline_score_zero_at=_f(thresholds["turnout_decline_score_zero_at"]),
        turnout_outside_band_share_good_max=_f(thresholds["turnout_outside_band_share_good_max"]),
        turnout_outside_band_share_zero_at=_f(thresholds["turnout_outside_band_share_zero_at"]),
        quality_component_weights=selection.get("quality_component_weights"),
    )

    assert len(scores) > 0
    assert int(scores.iloc[0]["design_id"]) == int(provenance["selected_design_id"])


def test_audit_script_contract_success_and_mismatch(tmp_path: Path, monkeypatch, capsys) -> None:
    bundle_dir = tmp_path / "doe_selection_bundle_v1"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    provenance = tmp_path / "doe_selection_provenance_v1.json"

    _write_json(
        bundle_dir / "doe_spec.json",
        {
            "primary_rule_name": "approval",
            "seeds": [1001, 1002, 1003],
        },
    )
    _write_json(
        bundle_dir / "doe_selection_spec.json",
        {
            "primary_rule_name": "approval",
            "required_primary_runs": 3,
            "weights": {"quality": 0.5, "stability": 0.5},
            "stage_weights": {"stage_a": 1.0},
            "thresholds": {
                "min_winner_entropy_norm": 0.1,
                "min_competitive_step_share": 0.1,
                "puzzle_dominance_share_score_low": 0.2,
                "puzzle_dominance_share_score_high": 0.8,
                "turnout_start_score_low": 0.2,
                "turnout_start_score_high": 0.8,
                "turnout_end_score_low": 0.2,
                "turnout_end_score_high": 0.8,
                "turnout_drop_score_good_max": 0.2,
                "turnout_drop_score_zero_at": 0.9,
                "turnout_decline_score_good_max": 0.2,
                "turnout_decline_score_zero_at": 0.9,
                "turnout_outside_band_share_good_max": 0.1,
                "turnout_outside_band_share_zero_at": 0.8,
            },
            "quality_component_weights": {"qg_alignment": 1.0},
        },
    )
    _write_csv(
        bundle_dir / "doe_run_features.csv",
        pd.DataFrame(
            [
                {"design_id": 42, "rule_idx": 1, "turnout_mean": 0.51},
                {"design_id": 7, "rule_idx": 1, "turnout_mean": 0.49},
            ]
        ),
    )
    _write_csv(
        bundle_dir / "doe_design_scores.csv",
        pd.DataFrame(
            [
                {"design_id": 42, "score_total": 0.91},
                {"design_id": 7, "score_total": 0.77},
            ]
        ),
    )
    _write_json(provenance, {"selected_design_id": 42})

    observed: dict[str, object] = {}

    def _fake_score_designs(df, **kwargs):  # noqa: ANN001
        observed["rows"] = len(df)
        observed["required_primary_runs"] = kwargs.get("required_primary_runs")
        observed["primary_rule_name"] = kwargs.get("primary_rule_name")
        return pd.DataFrame(
            [
                {"design_id": 42, "score_total": 0.91},
                {"design_id": 7, "score_total": 0.77},
            ]
        )

    monkeypatch.setattr(audit_lock, "score_designs", _fake_score_designs)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_doe_design_lock.py",
            "--bundle-dir",
            str(bundle_dir),
            "--provenance",
            str(provenance),
        ],
    )
    audit_lock.main()
    out = capsys.readouterr().out
    assert "[OK] DOE lock audit passed." in out
    assert observed["rows"] == 2
    assert observed["required_primary_runs"] == 3
    assert observed["primary_rule_name"] == "approval"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_doe_design_lock.py",
            "--bundle-dir",
            str(bundle_dir),
            "--provenance",
            str(provenance),
            "--expected-design-id",
            "7",
        ],
    )
    try:
        audit_lock.main()
    except ValueError as exc:
        assert "Design-lock audit failed" in str(exc)
    else:
        raise AssertionError("Expected explicit mismatch error for wrong expected design id.")

    def _empty_score_designs(_df, **_kwargs):  # noqa: ANN001
        return pd.DataFrame(columns=["design_id", "score_total"])

    monkeypatch.setattr(audit_lock, "score_designs", _empty_score_designs)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_doe_design_lock.py",
            "--bundle-dir",
            str(bundle_dir),
            "--provenance",
            str(provenance),
        ],
    )
    try:
        audit_lock.main()
    except ValueError as exc:
        assert "Recomputed ranking is empty." in str(exc)
    else:
        raise AssertionError("Expected explicit empty-ranking error.")

    def _mismatched_score_designs(_df, **_kwargs):  # noqa: ANN001
        return pd.DataFrame([{"design_id": 42, "score_total": 0.01}])

    monkeypatch.setattr(audit_lock, "score_designs", _mismatched_score_designs)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_doe_design_lock.py",
            "--bundle-dir",
            str(bundle_dir),
            "--provenance",
            str(provenance),
        ],
    )
    try:
        audit_lock.main()
    except ValueError as exc:
        assert "max |recomputed-published| exceeds tolerance" in str(exc)
    else:
        raise AssertionError("Expected explicit score-drift error.")


def test_audit_helpers_raise_for_invalid_inputs(tmp_path: Path) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[1,2,3]", encoding="utf-8")
    try:
        audit_lock._load_json(bad_json)
    except ValueError as exc:
        assert "Expected JSON object" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-object JSON payload.")

    missing_bundle_dir = tmp_path / "missing_bundle"
    missing_bundle_dir.mkdir()
    try:
        audit_lock._assert_required_bundle_files(missing_bundle_dir)
    except FileNotFoundError as exc:
        assert "Bundle missing required files" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing bundle files.")

    try:
        audit_lock._float("not-a-number", "threshold")
    except ValueError as exc:
        assert "Invalid numeric value for 'threshold'" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid numeric conversion.")
