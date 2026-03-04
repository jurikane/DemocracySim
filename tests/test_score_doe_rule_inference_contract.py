from __future__ import annotations

from pathlib import Path
import json

from scripts.score_doe import _infer_rules_from_doe_spec


def test_infer_rules_from_doe_spec_prefers_spec_values(tmp_path: Path) -> None:
    root = tmp_path / "doe_20990101_000003"
    root.mkdir(parents=True, exist_ok=True)
    (root / "doe_spec.json").write_text(
        json.dumps({"primary_rule_name": "approval", "robust_rule_name": "majority"}),
        encoding="utf-8",
    )

    primary, robust = _infer_rules_from_doe_spec(root)
    assert primary == "approval"
    assert robust == "majority"


def test_infer_rules_from_doe_spec_uses_fallback_defaults_when_spec_missing(tmp_path: Path) -> None:
    root = tmp_path / "doe_20990101_000004"
    root.mkdir(parents=True, exist_ok=True)

    primary, robust = _infer_rules_from_doe_spec(root)
    assert primary == "approval"
    assert robust == "utilitarian"
