from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import traceback
from typing import Any
import numpy as np
from tqdm import tqdm

from scripts.run_headless import run_once
from src.analysis.doe_runner import (
    apply_doe_overrides,
    build_run_plan,
    get_doe_profile,
    list_doe_profiles,
    sample_design_points,
    select_stratified_seeds,
    write_design_manifest,
    write_run_manifest,
    write_seed_selection_manifest,
)
from src.config.loader import load_config, resolve_output_dir


DOE_PROFILE_CHOICES = list(list_doe_profiles())


def _parse_seed_list(raw: str) -> list[int]:
    vals = [s.strip() for s in raw.split(",") if s.strip() != ""]
    if not vals:
        raise ValueError("Seed list is empty.")
    out: list[int] = []
    for s in vals:
        out.append(int(s))
    return out


def execute_run_plan(
    *,
    cfg,
    plan,
    run_once_fn=run_once,
    continue_on_error: bool = False,
    resume: bool = False,
    frozen_model: dict | None = None,
    frozen_simulation: dict | None = None,
) -> dict[str, int]:
    succeeded = 0
    failed = 0
    skipped = 0
    for task in tqdm(plan, desc="DOE runs"):
        if resume and _run_outputs_complete(task.out_dir):
            skipped += 1
            continue
        cfg_run = apply_doe_overrides(
            cfg,
            params=task.params,
            rule_idx=task.rule_idx,
            base_seed=task.seed,
            frozen_model=frozen_model,
            frozen_simulation=frozen_simulation,
        )
        task.out_dir.mkdir(parents=True, exist_ok=True)
        try:
            # run_id fixed to 0; base_seed controls actual model seed.
            run_once_fn(run_id=0, cfg=cfg_run, out_dir=task.out_dir)
            succeeded += 1
        except Exception:
            failed += 1
            if not continue_on_error:
                raise
            print(f"[DOE][WARN] run failed: design={task.design_id} seed={task.seed} rule={task.rule_idx}")
            print(traceback.format_exc())
    return {"succeeded": int(succeeded), "failed": int(failed), "skipped": int(skipped)}


def _run_outputs_complete(run_dir: Path) -> bool:
    required = [
        run_dir / "meta.yaml",
        run_dir / "static.json",
        run_dir / "steps.parquet",
        run_dir / "area_steps.parquet",
        run_dir / "agents.parquet",
    ]
    for p in required:
        if (not p.exists()) or (not p.is_file()):
            return False
        try:
            if p.stat().st_size <= 0:
                return False
        except OSError:
            return False
    return True


def _read_design_points(points_path: Path) -> list[dict[str, float]]:
    by_design_id: dict[int, dict[str, float]] = {}
    with points_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "design_id" not in reader.fieldnames:
            raise ValueError(f"Invalid DOE points manifest (missing design_id): {points_path}")
        for row in reader:
            did = int(row["design_id"])
            params: dict[str, float] = {}
            for k, v in row.items():
                if k == "design_id":
                    continue
                params[str(k)] = float(v)
            by_design_id[did] = params
    return [by_design_id[i] for i in sorted(by_design_id.keys())]


def _load_resume_context(out_root: Path) -> dict[str, Any] | None:
    spec_path = out_root / "doe_spec.json"
    points_path = out_root / "doe_design_points.csv"
    if not (spec_path.exists() and points_path.exists()):
        return None

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    design_points = _read_design_points(points_path)
    ranges_raw = dict(spec.get("ranges", {}))
    ranges = {str(k): (float(v[0]), float(v[1])) for k, v in ranges_raw.items()}
    return {
        "profile_name": str(spec.get("profile", "")),
        "design_points": design_points,
        "seeds": [int(s) for s in list(spec["seeds"])],
        "primary_rule_idx": int(spec["primary_rule_idx"]),
        "robust_rule_idx": int(spec["robust_rule_idx"]),
        "include_robustness": bool(spec["include_robustness"]),
        "robust_every": int(spec["robust_every"]),
        "ranges": ranges,
        "frozen_model": dict(spec.get("frozen_model", {})),
        "frozen_simulation": dict(spec.get("frozen_simulation", {})),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run phase-1 DOE screening (approval primary + utilitarian robustness).")
    parser.add_argument(
        "--config",
        "-c",
        default="doe.yaml",
        help="Config path or config name under configs/ (default: doe.yaml)",
    )
    parser.add_argument("--points", type=int, default=48, help="Number of DOE design points.")
    parser.add_argument(
        "--seed-mode",
        choices=["fixed", "stratified"],
        default="fixed",
        help="Seed selection mode. 'fixed' uses --seeds. 'stratified' selects spread-out seeds from candidate pool.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="101,202,303",
        help="Comma-separated fixed seeds for screening runs (used in --seed-mode fixed).",
    )
    parser.add_argument(
        "--seed-target",
        type=int,
        default=3,
        help="Number of seeds to select when --seed-mode stratified.",
    )
    parser.add_argument(
        "--seed-candidates",
        type=str,
        default="",
        help="Optional comma-separated candidate seeds for stratified mode. If omitted, range start/count is used.",
    )
    parser.add_argument(
        "--seed-candidate-start",
        type=int,
        default=100,
        help="Start value for candidate seed range in stratified mode.",
    )
    parser.add_argument(
        "--seed-candidate-count",
        type=int,
        default=40,
        help="Candidate seed count for stratified mode when --seed-candidates is not provided.",
    )
    parser.add_argument(
        "--seed-probe-rule-idx",
        type=int,
        default=None,
        help="Rule idx used for stratified seed probing (default: primary-rule-idx).",
    )
    parser.add_argument("--doe-seed", type=int, default=7, help="RNG seed used for sampling DOE points.")
    parser.add_argument(
        "--doe-profile",
        type=str,
        default="phase1",
        choices=DOE_PROFILE_CHOICES,
        help="DOE profile defining ranges + frozen settings.",
    )
    parser.add_argument("--primary-rule-idx", type=int, default=1, help="Primary screening rule (default=1 approval).")
    parser.add_argument("--robust-rule-idx", type=int, default=2, help="Robustness rule (default=2 utilitarian).")
    parser.add_argument("--no-robustness", action="store_true", help="Disable robustness runs.")
    parser.add_argument("--robust-every", type=int, default=1, help="Run robustness on every Nth design point.")
    parser.add_argument("--out-root", type=Path, default=None, help="Output root. Default: <output>/doe_<timestamp>")
    parser.add_argument("--dry-run", action="store_true", help="Plan DOE and write manifests, but do not execute runs.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining DOE runs if a run fails (warnings printed).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs that already look complete in --out-root (resume interrupted DOE).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    profile = get_doe_profile(str(args.doe_profile))
    ranges = dict(profile["ranges"])
    frozen_model = dict(profile["frozen_model"])
    frozen_simulation = dict(profile["frozen_simulation"])

    if args.out_root is None:
        base = resolve_output_dir(cfg)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = base / f"doe_{ts}"
    else:
        out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    resume_context = _load_resume_context(out_root) if bool(args.resume) else None
    using_existing_manifests = resume_context is not None
    if using_existing_manifests:
        design_points = list(resume_context["design_points"])
        seeds = list(resume_context["seeds"])
        primary_rule_idx = int(resume_context["primary_rule_idx"])
        robust_rule_idx = int(resume_context["robust_rule_idx"])
        include_robustness = bool(resume_context["include_robustness"])
        robust_every = int(resume_context["robust_every"])
        ranges = dict(resume_context["ranges"])
        frozen_model = dict(resume_context["frozen_model"])
        frozen_simulation = dict(resume_context["frozen_simulation"])
        print(f"[DOE][RESUME] Reusing existing manifests from {out_root}")
        prev_profile = str(resume_context.get("profile_name", ""))
        if prev_profile and prev_profile != str(args.doe_profile):
            print(
                f"[DOE][RESUME][WARN] Existing profile '{prev_profile}' differs from "
                f"CLI '--doe-profile {args.doe_profile}'. Existing manifest wins."
            )
    else:
        if args.seed_mode == "fixed":
            seeds = _parse_seed_list(args.seeds)
        else:
            if args.seed_candidates.strip():
                candidate_seeds = _parse_seed_list(args.seed_candidates)
            else:
                start = int(args.seed_candidate_start)
                count = int(args.seed_candidate_count)
                if count <= 0:
                    raise ValueError("--seed-candidate-count must be >= 1")
                candidate_seeds = list(range(start, start + count))
            probe_rule_idx = int(args.seed_probe_rule_idx) if args.seed_probe_rule_idx is not None else int(args.primary_rule_idx)
            seeds = select_stratified_seeds(
                cfg,
                target_count=int(args.seed_target),
                candidate_seeds=candidate_seeds,
                probe_rule_idx=probe_rule_idx,
                ranges=ranges,
                frozen_model=frozen_model,
                frozen_simulation=frozen_simulation,
            )

        primary_rule_idx = int(args.primary_rule_idx)
        robust_rule_idx = int(args.robust_rule_idx)
        include_robustness = not bool(args.no_robustness)
        robust_every = int(args.robust_every)

        if args.seed_mode == "fixed":
            write_seed_selection_manifest(
                out_root=out_root,
                mode="fixed",
                selected_seeds=seeds,
                candidate_seeds=seeds,
                probe_rule_idx=None,
            )
        else:
            write_seed_selection_manifest(
                out_root=out_root,
                mode="stratified",
                selected_seeds=seeds,
                candidate_seeds=candidate_seeds,
                probe_rule_idx=probe_rule_idx,
            )

        rng = np.random.default_rng(int(args.doe_seed))
        design_points = sample_design_points(num_points=int(args.points), rng=rng, ranges=ranges)
        write_design_manifest(
            out_root=out_root,
            design_points=design_points,
            seeds=seeds,
            primary_rule_idx=primary_rule_idx,
            robust_rule_idx=robust_rule_idx,
            include_robustness=include_robustness,
            robust_every=robust_every,
            ranges=ranges,
            frozen_model=frozen_model,
            frozen_simulation=frozen_simulation,
            profile_name=str(args.doe_profile),
        )

    plan = build_run_plan(
        out_root=out_root,
        design_points=design_points,
        seeds=seeds,
        primary_rule_idx=primary_rule_idx,
        robust_rule_idx=robust_rule_idx,
        include_robustness=include_robustness,
        robust_every=robust_every,
    )
    if not using_existing_manifests:
        write_run_manifest(out_root=out_root, plan=plan)

    started_at = datetime.now()
    print(f"DOE root: {out_root}")
    print(f"DOE started at: {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Design points: {len(design_points)} | Seeds: {len(seeds)} | Planned runs: {len(plan)}")

    if args.dry_run:
        print("DOE dry-run complete (no runs executed).")
        return

    summary = execute_run_plan(
        cfg=cfg,
        plan=plan,
        run_once_fn=run_once,
        continue_on_error=bool(args.continue_on_error),
        resume=bool(args.resume),
        frozen_model=frozen_model,
        frozen_simulation=frozen_simulation,
    )
    print(
        f"DOE summary: succeeded={summary['succeeded']} "
        f"skipped={summary.get('skipped', 0)} failed={summary['failed']}"
    )
    if summary["failed"] > 0 and bool(args.continue_on_error):
        print("[DOE][WARN] Some runs failed; inspect logs above.")

    finished_at = datetime.now()
    print("DOE finished:", out_root)
    print(f"DOE finished at: {finished_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"DOE duration: {finished_at - started_at}")


if __name__ == "__main__":
    main()
