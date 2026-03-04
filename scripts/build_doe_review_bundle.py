from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.doe_review_bundle import build_doe_review_bundle
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build consolidated DOE review bundle (top/mid/bottom run packets).")
    parser.add_argument("--doe-root", type=Path, default=None, help="DOE root directory (default: latest data/simulation_output/doe_*)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Bundle output directory (default: <DOE-root>/doe_score_output)")
    parser.add_argument("--per-bucket", type=int, default=3, help="Representative runs per bucket (top/mid/bottom).")
    parser.add_argument("--rule-name", type=str, default="approval", help="Rule used to select representative runs.")
    parser.add_argument(
        "--summary-profile",
        choices=(SUMMARY_PROFILE_FULL, SUMMARY_PROFILE_DEBUG_DOE_COMPACT, SUMMARY_PROFILE_THESIS_CORE),
        default=SUMMARY_PROFILE_DEBUG_DOE_COMPACT,
        help="Summary profile used for run PDFs in the bundle.",
    )
    parser.add_argument(
        "--skip-summary-render",
        action="store_true",
        help="Do not render per-run summary PDFs (metadata/overview still produced).",
    )
    parser.add_argument(
        "--allow-fallback-scores",
        action="store_true",
        help=(
            "Allow heuristic ranking from doe_run_features.csv when doe_design_scores.csv is empty/non-mergeable. "
            "Default is fail-fast to surface scoring problems."
        ),
    )
    args = parser.parse_args()

    root = _resolve_default_doe_root() if args.doe_root is None else Path(args.doe_root)
    out = build_doe_review_bundle(
        doe_root=root,
        out_dir=Path(args.out_dir) if args.out_dir is not None else None,
        per_bucket=int(args.per_bucket),
        rule_name=str(args.rule_name),
        summary_profile=str(args.summary_profile),
        render_summaries=not bool(args.skip_summary_render),
        allow_fallback_scores=bool(args.allow_fallback_scores),
    )
    print(f"DOE root: {root}")
    print(f"Wrote: {out.bundle_root}")
    print(f"Wrote: {out.queue_csv}")
    print(f"Wrote: {out.bucket_manifest_csv}")
    print(f"Wrote: {out.knob_ranges_csv}")
    print(f"Wrote: {out.knob_correlations_csv}")
    print(f"Wrote: {out.knob_effects_csv}")
    print(f"Wrote: {out.analysis_summary_pdf}")


if __name__ == "__main__":
    main()
