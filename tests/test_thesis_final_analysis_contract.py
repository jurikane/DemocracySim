from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.final_thesis_analysis import _normalize_rule_name


PACKAGE_ROOT = Path("artifacts/thesis_analysis_v1")
DERIVED_DIR = PACKAGE_ROOT / "derived"
TABLES_DIR = PACKAGE_ROOT / "tables"
FIGURES_DIR = PACKAGE_ROOT / "figures"

pytestmark = pytest.mark.skipif(
    not PACKAGE_ROOT.exists(),
    reason="requires local thesis analysis package payload under artifacts/thesis_analysis_v1",
)


def test_final_analysis_package_batch_and_endpoint_contract() -> None:
    run_level = pd.read_csv(DERIVED_DIR / "run_level_endpoint_summary.csv")

    assert len(run_level) == 1050
    assert set(run_level["quality_target_mode"].astype(str)) == {"puzzle"}
    assert run_level[
        [
            "turnout_mean",
            "gini_assets_mean",
            "gini_dissatisfaction_mean",
            "quality_distance_mean",
        ]
    ].notna().all().all()

    rule_counts = run_level["rule_name"].value_counts().sort_index().to_dict()
    assert rule_counts == {
        "approval": 50,
        "borda": 200,
        "plurality": 200,
        "random": 200,
        "schulze": 200,
        "utilitarian": 200,
    }

    grouped = (
        run_level.groupby("rule_name", sort=True)["rule_group"]
        .agg(lambda s: set(s.astype(str)))
        .to_dict()
    )
    assert grouped == {
        "approval": {"context"},
        "borda": {"canonical"},
        "plurality": {"reference"},
        "random": {"reference"},
        "schulze": {"canonical"},
        "utilitarian": {"canonical"},
    }


def test_final_analysis_tables_figures_and_scope_contract() -> None:
    t3 = pd.read_csv(TABLES_DIR / "T3_endpoint_summaries_per_rule.csv")
    t4 = pd.read_csv(TABLES_DIR / "T4_canonical_confirmatory_results.csv")
    t5 = pd.read_csv(TABLES_DIR / "T5_reference_family_results.csv")
    t6 = pd.read_csv(TABLES_DIR / "T6_robustness_summaries.csv")

    assert len(t3) == 6
    assert len(t4) == 12
    assert len(t5) == 24
    assert len(t6) == 12

    assert set(t3["rule_name"].astype(str)) == {
        "approval",
        "borda",
        "plurality",
        "random",
        "schulze",
        "utilitarian",
    }
    assert "approval" not in set(t4["left_rule"].astype(str)) | set(t4["right_rule"].astype(str))
    assert "approval" not in set(t5["left_rule"].astype(str)) | set(t5["right_rule"].astype(str))

    assert set(t4["contrast"].astype(str)) == {
        "utilitarian - borda",
        "utilitarian - schulze",
        "borda - schulze",
    }
    assert set(t5["contrast"].astype(str)) == {
        "utilitarian - plurality",
        "utilitarian - random",
        "borda - plurality",
        "borda - random",
        "schulze - plurality",
        "schulze - random",
    }

    expected_figures = {
        "F1_primary_metric_trajectories.pdf",
        "F1_primary_metric_trajectories.png",
        "F2_paired_endpoint_comparisons.pdf",
        "F2_paired_endpoint_comparisons.png",
        "F3_canonical_confirmatory_effect_forest.pdf",
        "F3_canonical_confirmatory_effect_forest.png",
        "F4_reference_family_effect_panel.pdf",
        "F4_reference_family_effect_panel.png",
    }
    assert {p.name for p in FIGURES_DIR.iterdir() if p.is_file()} == expected_figures


def test_final_analysis_provenance_docs_and_t1_contract() -> None:
    provenance = json.loads((PACKAGE_ROOT / "analysis_provenance.json").read_text(encoding="utf-8"))
    t1 = pd.read_csv(TABLES_DIR / "T1_frozen_run_protocol_provenance.csv")
    freeze = json.loads(Path("configs/thesis/freeze_provenance_v1.json").read_text(encoding="utf-8"))
    docs_page = Path("docs/technical/final_analysis_package.md").read_text(encoding="utf-8")
    package_readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")

    required_keys = {
        "version",
        "generated_at_utc",
        "source_run_root",
        "source_manifest",
        "freeze_provenance",
        "protocol_path",
        "git_head",
        "analysis_seed",
        "permutation_draws",
        "bootstrap_reps",
        "expected_runs_total",
        "rule_counts_expected",
        "known_caveats",
        "realized_runs_total",
        "rule_counts_realized",
        "generated_artifacts",
    }
    assert required_keys.issubset(provenance)
    assert provenance["version"] == "thesis_analysis_v1"
    assert provenance["source_run_root"] == "data/simulation_output/thesis_final_runs_v1"
    assert provenance["source_manifest"] == "configs/thesis/final_run_manifest_v1.csv"
    assert provenance["freeze_provenance"] == "configs/thesis/freeze_provenance_v1.json"
    assert provenance["protocol_path"] == "configs/thesis/thesis_analysis_protocol_v1.md"
    assert provenance["analysis_seed"] == 20260311
    assert provenance["permutation_draws"] == 100000
    assert provenance["bootstrap_reps"] == 10000
    assert provenance["expected_runs_total"] == 1050
    assert provenance["realized_runs_total"] == 1050
    assert len(provenance["known_caveats"]) == 3
    assert len(provenance["generated_artifacts"]) == 19

    caveat_ids = {item["id"] for item in provenance["known_caveats"]}
    assert caveat_ids == {
        "seed_level_config_used",
        "stale_params_json_rule_idx",
        "asset_scale_explosive_but_finite",
    }

    t1_map = dict(zip(t1["key"].astype(str), t1["value"].astype(str), strict=True))
    assert t1_map["doe_batch_id"] == "DOE-20260302-175623"
    assert t1_map["selected_design_id"] == "149"
    assert t1_map["schema_version"] == "schema-v2"
    assert t1_map["expected_runs"] == "1050"
    assert t1_map["realized_runs"] == "1050"
    assert t1_map["steps_per_run"] == "250"

    git_head = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
    )
    .stdout.strip()
    )
    assert provenance["git_head"] == git_head
    assert t1_map["git_head"] == str(freeze["git_head"])

    assert "raw final-run directories remain outside Git" in docs_page
    assert "tables/free-rider.csv" in docs_page
    assert "tables/free-rider-support.csv" in docs_page
    assert "free-rider.csv" in package_readme
    assert "free-rider-support.csv" in package_readme


def test_final_analysis_accepts_frozen_pre_rename_rule_label() -> None:
    assert _normalize_rule_name("majority") == "plurality"
    assert _normalize_rule_name("plurality") == "plurality"
