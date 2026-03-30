from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analysis.final_thesis_analysis import (
    AnalysisSettings,
    DEFAULT_ANALYSIS_SEED,
    DEFAULT_BOOTSTRAP_REPS,
    DEFAULT_FINAL_MANIFEST,
    DEFAULT_FINAL_RUN_ROOT,
    DEFAULT_FREEZE_PROVENANCE,
    DEFAULT_PACKAGE_OUT_DIR,
    DEFAULT_PERMUTATION_DRAWS,
    DEFAULT_PROTOCOL,
    build_full_analysis_package,
    resolve_analysis_layout,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the tracked thesis-final analysis package scaffold.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_FINAL_RUN_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_FINAL_MANIFEST)
    parser.add_argument("--freeze-provenance", type=Path, default=DEFAULT_FREEZE_PROVENANCE)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PACKAGE_OUT_DIR)
    parser.add_argument("--analysis-seed", type=int, default=DEFAULT_ANALYSIS_SEED)
    parser.add_argument("--permutation-draws", type=int, default=DEFAULT_PERMUTATION_DRAWS)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source, output = resolve_analysis_layout(
        run_root=args.run_root,
        manifest=args.manifest,
        freeze_provenance=args.freeze_provenance,
        protocol=args.protocol,
        out_dir=args.out_dir,
    )
    settings = AnalysisSettings(
        analysis_seed=int(args.analysis_seed),
        permutation_draws=int(args.permutation_draws),
        bootstrap_reps=int(args.bootstrap_reps),
    )

    resolved = {
        "run_root": source.run_root.as_posix(),
        "manifest": source.manifest.as_posix(),
        "freeze_provenance": source.freeze_provenance.as_posix(),
        "protocol": source.protocol.as_posix(),
        "out_dir": output.out_dir.as_posix(),
        "analysis_seed": settings.analysis_seed,
        "permutation_draws": settings.permutation_draws,
        "bootstrap_reps": settings.bootstrap_reps,
    }
    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        return

    provenance = build_full_analysis_package(source=source, output=output, settings=settings)
    print(json.dumps(
        {
            "created": {
                "out_dir": output.out_dir.as_posix(),
                "derived_dir": output.derived_dir.as_posix(),
                "tables_dir": output.tables_dir.as_posix(),
                "figures_dir": output.figures_dir.as_posix(),
                "readme": output.readme_path.as_posix(),
                "analysis_provenance": output.provenance_json_path.as_posix(),
                "run_level_endpoint_summary": output.run_level_csv_path.as_posix(),
                "rule_step_primary_summary": output.rule_step_csv_path.as_posix(),
                "tables_dir": output.tables_dir.as_posix(),
                "figures_dir": output.figures_dir.as_posix(),
            },
            "provenance_version": provenance["version"],
            "realized_runs_total": provenance["realized_runs_total"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
