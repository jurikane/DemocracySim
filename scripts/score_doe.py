from __future__ import annotations

import argparse
from pathlib import Path
import json

from src.analysis.doe_review_bundle import build_doe_review_bundle
from src.analysis.doe_scoring import analyze_doe_root
from src.analysis.summary_tooling import (
    SUMMARY_PROFILE_DEBUG_DOE_COMPACT,
    SUMMARY_PROFILE_FULL,
    SUMMARY_PROFILE_THESIS_CORE,
)


def _resolve_default_doe_root() -> Path:
    base = Path("data") / "simulation_output"
    candidates = sorted([p for p in base.glob("doe_*") if p.is_dir()])
    if not candidates:
        raise FileNotFoundError("No DOE directories found under data/simulation_output (expected doe_*).")
    return candidates[-1]


def _infer_rules_from_doe_spec(root: Path) -> tuple[str, str]:
    spec_path = root / "doe_spec.json"
    default_primary = "approval"
    default_robust = "utilitarian"
    if not spec_path.exists():
        return default_primary, default_robust
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return default_primary, default_robust
    if not isinstance(spec, dict):
        return default_primary, default_robust
    primary = str(spec.get("primary_rule_name", default_primary) or default_primary)
    robust = str(spec.get("robust_rule_name", default_robust) or default_robust)
    return primary, robust


def main() -> None:
    parser = argparse.ArgumentParser(description="Score DOE outputs (hard gates + weighted ranking).")
    parser.add_argument("--doe-root", type=Path, default=None, help="DOE root directory (default: latest data/simulation_output/doe_*)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: DOE root)")
    parser.add_argument("--burn-in-steps", type=int, default=0, help="Warm-up exclusion window for lock-in/chaos metrics (analysis-only).")
    parser.add_argument(
        "--primary-rule",
        type=str,
        default=None,
        help="Primary rule folder name (without 'rule_'). Default: infer from doe_spec.json, fallback 'approval'.",
    )
    parser.add_argument(
        "--robust-rule",
        type=str,
        default=None,
        help="Robustness rule folder name (without 'rule_'). Default: infer from doe_spec.json, fallback 'utilitarian'.",
    )
    parser.add_argument(
        "--objective-config",
        type=Path,
        default=Path("configs") / "doe_selection_objective_thesis_recovery_v1.json",
        help="Selection objective JSON (thresholds/weights/stage-weights/strict-completeness).",
    )
    parser.add_argument(
        "--allow-incomplete-designs",
        action="store_true",
        help="Disable strict completeness filter (default is strict filtering by expected seed coverage).",
    )
    parser.add_argument(
        "--skip-review-bundle",
        action="store_true",
        help="Do not build the post-scoring DOE review bundle folder.",
    )
    parser.add_argument(
        "--bundle-out-dir",
        type=Path,
        default=None,
        help="Review bundle output directory (default: <DOE-root>/doe_score_output).",
    )
    parser.add_argument(
        "--bundle-per-bucket",
        type=int,
        default=3,
        help="Runs per bucket (top/mid/bottom) in review bundle.",
    )
    parser.add_argument(
        "--bundle-rule-name",
        type=str,
        default=None,
        help="Rule name used for selecting representative runs in bundle (default: --primary-rule).",
    )
    parser.add_argument(
        "--bundle-summary-profile",
        choices=(SUMMARY_PROFILE_FULL, SUMMARY_PROFILE_DEBUG_DOE_COMPACT, SUMMARY_PROFILE_THESIS_CORE),
        default=SUMMARY_PROFILE_DEBUG_DOE_COMPACT,
        help="Summary profile used for per-run PDFs in review bundle.",
    )
    parser.add_argument(
        "--bundle-allow-fallback-scores",
        action="store_true",
        help=(
            "Allow heuristic ranking from doe_run_features.csv when doe_design_scores.csv is empty/non-mergeable "
            "while building the review bundle. Default is fail-fast."
        ),
    )
    args = parser.parse_args()

    root = _resolve_default_doe_root() if args.doe_root is None else Path(args.doe_root)
    inferred_primary, inferred_robust = _infer_rules_from_doe_spec(root)
    primary_rule = str(args.primary_rule or inferred_primary)
    robust_rule = str(args.robust_rule or inferred_robust)
    out = analyze_doe_root(
        root,
        out_dir=args.out_dir,
        burn_in_steps=int(args.burn_in_steps),
        primary_rule_name=primary_rule,
        robust_rule_name=robust_rule,
        objective_config_path=Path(args.objective_config) if args.objective_config else None,
        strict_completeness=(
            False if bool(args.allow_incomplete_designs) else None
        ),
    )
    print(f"DOE root: {root}")
    print(f"Rules: primary={primary_rule}, robust={robust_rule}")
    print(f"Wrote: {out['run_features_csv']}")
    print(f"Wrote: {out['design_scores_csv']}")
    print(f"Wrote: {out['selection_spec_json']}")
    print(f"Wrote: {out['top_designs_json']}")

    if not bool(args.skip_review_bundle):
        required = [
            root / "doe_design_scores.csv",
            root / "doe_run_features.csv",
            root / "doe_design_points.csv",
        ]
        missing = [p.name for p in required if not p.exists()]
        if missing:
            print(
                "Skipping review bundle: missing required DOE files: "
                + ", ".join(sorted(missing))
            )
        else:
            bundle = build_doe_review_bundle(
                doe_root=root,
                out_dir=Path(args.bundle_out_dir) if args.bundle_out_dir is not None else None,
                per_bucket=int(args.bundle_per_bucket),
                rule_name=str(args.bundle_rule_name or primary_rule),
                summary_profile=str(args.bundle_summary_profile),
                render_summaries=True,
                allow_fallback_scores=bool(args.bundle_allow_fallback_scores),
            )
            print(f"Wrote: {bundle.bundle_root}")
            print(f"Wrote: {bundle.queue_csv}")
            print(f"Wrote: {bundle.bucket_manifest_csv}")
            print(f"Wrote: {bundle.knob_ranges_csv}")
            print(f"Wrote: {bundle.knob_correlations_csv}")
            print(f"Wrote: {bundle.knob_effects_csv}")
            print(f"Wrote: {bundle.analysis_summary_pdf}")


if __name__ == "__main__":
    main()
