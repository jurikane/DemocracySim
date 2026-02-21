from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.doe_scoring import analyze_doe_root


def _resolve_default_doe_root() -> Path:
    base = Path("data") / "simulation_output"
    candidates = sorted([p for p in base.glob("doe_*") if p.is_dir()])
    if not candidates:
        raise FileNotFoundError("No DOE directories found under data/simulation_output (expected doe_*).")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score DOE outputs (hard gates + weighted ranking).")
    parser.add_argument("--doe-root", type=Path, default=None, help="DOE root directory (default: latest data/simulation_output/doe_*)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: DOE root)")
    parser.add_argument("--burn-in-steps", type=int, default=0, help="Warm-up exclusion window for lock-in/chaos metrics (analysis-only).")
    parser.add_argument("--primary-rule", type=str, default="approval", help="Primary rule folder name (without 'rule_').")
    parser.add_argument("--robust-rule", type=str, default="utilitarian", help="Robustness rule folder name (without 'rule_').")
    parser.add_argument(
        "--objective-config",
        type=Path,
        default=Path("configs") / "doe_selection_objective_v1.json",
        help="Selection objective JSON (thresholds/weights/stage-weights/strict-completeness).",
    )
    parser.add_argument(
        "--allow-incomplete-designs",
        action="store_true",
        help="Disable strict completeness filter (default is strict filtering by expected seed coverage).",
    )
    args = parser.parse_args()

    root = _resolve_default_doe_root() if args.doe_root is None else Path(args.doe_root)
    out = analyze_doe_root(
        root,
        out_dir=args.out_dir,
        burn_in_steps=int(args.burn_in_steps),
        primary_rule_name=str(args.primary_rule),
        robust_rule_name=str(args.robust_rule),
        objective_config_path=Path(args.objective_config) if args.objective_config else None,
        strict_completeness=(
            False if bool(args.allow_incomplete_designs) else None
        ),
    )
    print(f"DOE root: {root}")
    print(f"Wrote: {out['run_features_csv']}")
    print(f"Wrote: {out['design_scores_csv']}")
    print(f"Wrote: {out['selection_spec_json']}")
    print(f"Wrote: {out['top_designs_json']}")


if __name__ == "__main__":
    main()
