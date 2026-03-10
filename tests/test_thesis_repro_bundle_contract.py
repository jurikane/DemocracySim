from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from scripts.repro import restamp_freeze_provenance as restamp
from scripts.repro import verify_thesis_repro_bundle as verify_bundle


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def test_freeze_provenance_hash_block_matches_files() -> None:
    freeze_path = Path("configs/thesis/freeze_provenance_v1.json")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))

    hash_block = freeze["artifacts"]["sha256"]
    assert isinstance(hash_block, dict)
    assert hash_block

    for rel, expected in hash_block.items():
        p = Path(rel)
        assert p.exists(), f"Missing published artifact: {p}"
        assert _sha256(p) == str(expected), f"Hash mismatch for artifact: {p}"


def test_freeze_run_plan_counts_match_seed_contract() -> None:
    freeze = json.loads(Path("configs/thesis/freeze_provenance_v1.json").read_text(encoding="utf-8"))
    seeds = json.loads(Path("configs/thesis/final_seed_list_v1.json").read_text(encoding="utf-8"))

    expected_total = len(seeds["main_rules"]) * len(seeds["S_main"]) + len(seeds["S_approval"])
    assert freeze["run_plan_counts"]["expected_runs_total"] == expected_total
    assert freeze["run_plan_counts"]["main_rules"] == len(seeds["main_rules"])
    assert freeze["run_plan_counts"]["main_seeds"] == len(seeds["S_main"])
    assert freeze["run_plan_counts"]["approval_seeds"] == len(seeds["S_approval"])

    assert seeds["S_approval"] == seeds["S_main"][: len(seeds["S_approval"])]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_restamp_and_verify_bundle_roundtrip(tmp_path: Path, monkeypatch, capsys) -> None:
    design_id = 38
    run_root = str(tmp_path / "sim_output")

    final_model = tmp_path / "final_model_v1.yaml"
    final_seeds = tmp_path / "final_seed_list_v1.json"
    final_manifest = tmp_path / "final_run_manifest_v1.csv"
    selection_provenance = tmp_path / "doe_selection_provenance_v1.json"
    bundle_dir = tmp_path / "doe_selection_bundle_v1"
    freeze_out = tmp_path / "freeze_provenance_v1.json"

    model_payload = {"model": {"known_cells": 8, "num_steps": 10}}
    final_model.write_text(yaml.safe_dump(model_payload, sort_keys=False), encoding="utf-8")

    seed_payload = {
        "selected_design_id": design_id,
        "main_rules": [0, 2],
        "S_main": [101, 102],
        "S_approval": [101],
    }
    _write_json(final_seeds, seed_payload)

    _write_json(
        selection_provenance,
        {
            "doe_batch_id": "DOE-UNITTEST",
            "doe_profile": "freeze_contract",
            "selected_design_id": design_id,
        },
    )

    bundle_dir.mkdir(parents=True, exist_ok=True)
    _write_json(bundle_dir / "doe_spec.json", {"profile": "freeze_contract", "seeds": [101, 102]})
    _write_json(bundle_dir / "doe_selection_spec.json", {"weights": {"quality": 1.0}})
    _write_csv(bundle_dir / "doe_run_features.csv", pd.DataFrame([{"design_id": design_id, "rule_idx": 0}]))
    _write_csv(bundle_dir / "doe_design_scores.csv", pd.DataFrame([{"design_id": design_id, "score_total": 0.9}]))

    expected_rows = verify_bundle._build_expected_manifest_rows(
        model_params=model_payload["model"],
        design_id=design_id,
        rules_main=seed_payload["main_rules"],
        seeds_main=seed_payload["S_main"],
        approval_count=len(seed_payload["S_approval"]),
        run_root=run_root,
    )
    final_manifest.write_bytes(verify_bundle._manifest_rows_to_bytes(expected_rows))

    monkeypatch.setattr(restamp, "_git_head", lambda: "deadbeef")
    monkeypatch.setattr(restamp, "_git_dirty", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "restamp_freeze_provenance.py",
            "--out",
            str(freeze_out),
            "--final-model",
            str(final_model),
            "--final-seeds",
            str(final_seeds),
            "--final-manifest",
            str(final_manifest),
            "--doe-selection-provenance",
            str(selection_provenance),
            "--doe-bundle-dir",
            str(bundle_dir),
            "--release-tag",
            "thesis-freeze-v1",
        ],
    )
    restamp.main()
    assert freeze_out.exists()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_thesis_repro_bundle.py",
            "--freeze-provenance",
            str(freeze_out),
            "--final-model",
            str(final_model),
            "--final-seeds",
            str(final_seeds),
            "--final-manifest",
            str(final_manifest),
        ],
    )
    verify_bundle.main()
    out = capsys.readouterr().out
    assert "[OK] Hash block verified for published artifacts." in out
    assert "[OK] Deterministic manifest hash verified:" in out


def test_repro_helper_error_branches(tmp_path: Path, monkeypatch) -> None:
    def _raise_check_output(*_args, **_kwargs):  # noqa: ANN001
        raise FileNotFoundError

    monkeypatch.setattr(restamp.subprocess, "check_output", _raise_check_output)
    assert restamp._git_head() == "UNKNOWN"

    def _raise_check_call(*_args, **_kwargs):  # noqa: ANN001
        raise subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr(restamp.subprocess, "check_call", _raise_check_call)
    assert restamp._git_dirty() is True

    monkeypatch.setattr(restamp.subprocess, "check_call", lambda *_args, **_kwargs: 0)
    assert restamp._git_dirty() is False

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[1,2,3]", encoding="utf-8")
    try:
        restamp._load_json(bad_json)
    except ValueError as exc:
        assert "Expected JSON object" in str(exc)
    else:
        raise AssertionError("Expected ValueError for restamp non-object JSON.")

    try:
        verify_bundle._load_json(bad_json)
    except ValueError as exc:
        assert "Expected JSON object" in str(exc)
    else:
        raise AssertionError("Expected ValueError for verify non-object JSON.")

    d = 7
    assert verify_bundle._infer_run_root_from_manifest(f"design_{d:04d}/rule_x/seed_00001/run_0", d) == ""
    assert (
        verify_bundle._infer_run_root_from_manifest("no-design-marker/here", d)
        == "data/simulation_output/thesis_final_runs_v1"
    )


def test_repro_main_validation_branches(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "freeze_provenance_v1.json"
    final_model = tmp_path / "final_model_v1.yaml"
    final_seeds = tmp_path / "final_seed_list_v1.json"
    final_manifest = tmp_path / "final_run_manifest_v1.csv"
    sel_prov = tmp_path / "doe_selection_provenance_v1.json"
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    final_model.write_text(yaml.safe_dump({"model": {"x": 1}}, sort_keys=False), encoding="utf-8")
    _write_json(final_seeds, {"selected_design_id": 1, "main_rules": [0], "S_main": [1], "S_approval": [1]})
    _write_json(sel_prov, {"selected_design_id": 1})

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "restamp_freeze_provenance.py",
            "--out",
            str(out),
            "--final-model",
            str(final_model),
            "--final-seeds",
            str(final_seeds),
            "--final-manifest",
            str(final_manifest),
            "--doe-selection-provenance",
            str(sel_prov),
            "--doe-bundle-dir",
            str(bundle_dir),
        ],
    )
    try:
        restamp.main()
    except FileNotFoundError as exc:
        assert "Missing required input" in str(exc)
    else:
        raise AssertionError("Expected missing-input error from restamp.main().")

    final_manifest.write_text("run_index\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "restamp_freeze_provenance.py",
            "--out",
            str(out),
            "--final-model",
            str(final_model),
            "--final-seeds",
            str(final_seeds),
            "--final-manifest",
            str(final_manifest),
            "--doe-selection-provenance",
            str(sel_prov),
            "--doe-bundle-dir",
            str(bundle_dir),
        ],
    )
    try:
        restamp.main()
    except FileNotFoundError as exc:
        assert "No files found in DOE bundle dir" in str(exc)
    else:
        raise AssertionError("Expected empty-bundle error from restamp.main().")

    freeze = tmp_path / "freeze.json"
    _write_json(freeze, {"artifacts": {"sha256": {}}, "run_plan_counts": {"expected_runs_total": 0}})
    bad_model = tmp_path / "bad_model.yaml"
    bad_model.write_text(yaml.safe_dump(["not", "an", "object"]), encoding="utf-8")
    good_seeds = tmp_path / "good_seeds.json"
    _write_json(good_seeds, {"selected_design_id": 1, "main_rules": [0], "S_main": [1], "S_approval": [1]})
    empty_manifest = tmp_path / "empty_manifest.csv"
    empty_manifest.write_text(
        "run_index,design_id,rule_idx,rule_name,seed,family_role,out_dir,params_json,params_hash\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_thesis_repro_bundle.py",
            "--freeze-provenance",
            str(freeze),
            "--final-model",
            str(bad_model),
            "--final-seeds",
            str(good_seeds),
            "--final-manifest",
            str(empty_manifest),
        ],
    )
    try:
        verify_bundle.main()
    except ValueError as exc:
        assert "must parse to an object" in str(exc)
    else:
        raise AssertionError("Expected parse-shape error for final_model YAML.")

    missing_model_key = tmp_path / "missing_model_key.yaml"
    missing_model_key.write_text(yaml.safe_dump({"not_model": {}}, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_thesis_repro_bundle.py",
            "--freeze-provenance",
            str(freeze),
            "--final-model",
            str(missing_model_key),
            "--final-seeds",
            str(good_seeds),
            "--final-manifest",
            str(empty_manifest),
        ],
    )
    try:
        verify_bundle.main()
    except ValueError as exc:
        assert "missing object key: model" in str(exc)
    else:
        raise AssertionError("Expected missing-model-key error.")
