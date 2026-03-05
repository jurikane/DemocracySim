from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.doe_refine import build_inference_report


def _resolve_default_doe_root() -> Path:
    base = Path("data") / "simulation_output"
    candidates = sorted([p for p in base.glob("doe_*") if p.is_dir()])
    if not candidates:
        raise FileNotFoundError("No DOE directories found under data/simulation_output (expected doe_*).")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DOE inference artifacts (seed effects, nonlinear importance, interactions, CIs, Pareto).")
    parser.add_argument("--doe-root", type=Path, default=None, help="DOE root directory (default: latest data/simulation_output/doe_*)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: DOE root)")
    parser.add_argument("--bootstrap-reps", type=int, default=500, help="Bootstrap repetitions for CI estimation.")
    parser.add_argument("--random-seed", type=int, default=11, help="Random seed for inference bootstrap.")
    args = parser.parse_args()

    root = _resolve_default_doe_root() if args.doe_root is None else Path(args.doe_root)
    out = build_inference_report(
        root,
        out_dir=args.out_dir,
        bootstrap_reps=int(args.bootstrap_reps),
        random_seed=int(args.random_seed),
    )
    print(f"DOE root: {root}")
    print(f"Wrote: {out['inference_spec_json']}")
    print(f"Wrote: {out['seed_fixed_effects_csv']}")
    print(f"Wrote: {out['nonlinear_importance_csv']}")
    print(f"Wrote: {out['interaction_maps_csv']}")
    print(f"Wrote: {out['bootstrap_design_ci_csv']}")
    print(f"Wrote: {out['pareto_designs_csv']}")


if __name__ == "__main__":
    main()
