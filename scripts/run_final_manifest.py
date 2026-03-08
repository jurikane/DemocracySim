from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scripts.run_doe import _run_outputs_complete
from scripts.run_headless import run_once
from src.config.loader import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description="Run frozen thesis final-run manifest.")
    ap.add_argument("--config", type=Path, default=Path("configs/thesis/final_model_v1.yaml"))
    ap.add_argument("--manifest", type=Path, default=Path("configs/thesis/final_run_manifest_v1.csv"))
    ap.add_argument("--dry-run", action="store_true", help="Validate manifest/config and print planned run count only.")
    ap.add_argument("--max-runs", type=int, default=0, help="Optional cap for smoke tests (0 = all manifest rows).")
    ap.add_argument("--resume", action="store_true", help="Skip runs with complete artifacts.")
    ap.add_argument("--continue-on-error", action="store_true", help="Continue remaining runs after failures.")
    args = ap.parse_args()

    cfg = load_config(str(args.config))
    manifest = Path(args.manifest)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    succeeded = 0
    failed = 0
    skipped = 0

    with manifest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"rule_idx", "seed", "out_dir", "run_index"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Manifest missing required columns. Need: {sorted(required)}")
        rows = list(reader)
        cap = int(args.max_runs)
        if cap > 0:
            rows = rows[:cap]
        print(f"Planned rows to execute: {len(rows)}")
        if bool(args.dry_run):
            return
        for row in rows:
            out_dir = Path(str(row["out_dir"]))
            if bool(args.resume) and _run_outputs_complete(out_dir):
                skipped += 1
                continue
            cfg_run = cfg.model_copy(deep=True)
            cfg_run.model.rule_idx = int(row["rule_idx"])
            cfg_run.simulation.base_seed = int(row["seed"])
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                run_once(run_id=0, cfg=cfg_run, out_dir=out_dir)
                succeeded += 1
            except (RuntimeError, ValueError, TypeError, OSError) as exc:
                failed += 1
                print(
                    f"[FINAL][WARN] failed run_index={row.get('run_index','?')} "
                    f"rule_idx={row.get('rule_idx','?')} seed={row.get('seed','?')} err={exc}"
                )
                if not bool(args.continue_on_error):
                    raise

    print(f"Completed final manifest: succeeded={succeeded}, failed={failed}, skipped={skipped}")


if __name__ == "__main__":
    main()
