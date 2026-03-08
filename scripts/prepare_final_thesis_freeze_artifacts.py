from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
from typing import Any

import pandas as pd
import yaml

from src.analysis.doe_runner import rule_label
from src.config.schema import ModelConfig


INTEGER_MODEL_KEYS: set[str] = {
    "rule_idx",
    "distance_idx",
    "known_cells",
    "num_agents",
    "num_colors",
    "num_personality_groups",
    "height",
    "width",
    "num_areas",
    "av_area_height",
    "av_area_width",
    "color_patches_steps",
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _git_head() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return "UNKNOWN"


def _git_dirty() -> bool:
    try:
        subprocess.check_call(["git", "diff", "--quiet"])
        subprocess.check_call(["git", "diff", "--cached", "--quiet"])
        return False
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML object in: {path}")
    return raw


def _build_seed_list(*, seed_count: int, seed_rng: int, seed_min: int, seed_max: int) -> list[int]:
    if seed_count <= 0:
        raise ValueError("seed_count must be >= 1")
    n = seed_max - seed_min + 1
    if n < seed_count:
        raise ValueError(
            f"Seed range too small for requested seed_count: range_size={n}, seed_count={seed_count}"
        )
    rng = random.Random(int(seed_rng))
    pool = range(int(seed_min), int(seed_max) + 1)
    return [int(x) for x in rng.sample(pool, k=int(seed_count))]


def _write_manifest(
    *,
    out_csv: Path,
    run_root: Path,
    model_params: dict[str, Any],
    design_id: int,
    seeds_main: list[int],
    rules_main: list[int],
    approval_context_count: int,
) -> None:
    rows: list[dict[str, Any]] = []
    params_json = json.dumps(model_params, sort_keys=True, separators=(",", ":"))
    params_hash = hashlib.sha256(params_json.encode("utf-8")).hexdigest()

    for ridx in rules_main:
        rname = rule_label(int(ridx))
        for seed in seeds_main:
            out_dir = run_root / f"design_{int(design_id):04d}" / f"rule_{rname}" / f"seed_{int(seed):05d}" / "run_0"
            rows.append(
                {
                    "run_index": int(len(rows)),
                    "design_id": int(design_id),
                    "rule_idx": int(ridx),
                    "rule_name": str(rname),
                    "seed": int(seed),
                    "family_role": "main",
                    "out_dir": str(out_dir),
                    "params_json": params_json,
                    "params_hash": params_hash,
                }
            )

    for seed in seeds_main[: int(approval_context_count)]:
        ridx = 1
        rname = rule_label(int(ridx))
        out_dir = run_root / f"design_{int(design_id):04d}" / f"rule_{rname}" / f"seed_{int(seed):05d}" / "run_0"
        rows.append(
            {
                "run_index": int(len(rows)),
                "design_id": int(design_id),
                "rule_idx": int(ridx),
                "rule_name": str(rname),
                "seed": int(seed),
                "family_role": "approval_context",
                "out_dir": str(out_dir),
                "params_json": params_json,
                "params_hash": params_hash,
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run_index",
                "design_id",
                "rule_idx",
                "rule_name",
                "seed",
                "family_role",
                "out_dir",
                "params_json",
                "params_hash",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare frozen thesis final-run artifacts (model config, seeds, manifest, provenance).")
    ap.add_argument(
        "--source-doe-root",
        type=Path,
        default=Path("data/simulation_output/doe_20260302_175623"),
        help="Authoritative source DOE root.",
    )
    ap.add_argument("--design-id", type=int, default=149, help="Selected frozen design ID.")
    ap.add_argument("--base-config", type=Path, default=Path("configs/doe.yaml"), help="Base config used for visualization/output blocks.")
    ap.add_argument("--seed-count-main", type=int, default=200, help="S_main size.")
    ap.add_argument("--seed-count-approval", type=int, default=50, help="S_approval size (first N from S_main).")
    ap.add_argument("--seed-rng", type=int, default=20260308, help="RNG seed for deterministic matched-seed generation.")
    ap.add_argument("--seed-min", type=int, default=10000, help="Seed sampling lower bound (inclusive).")
    ap.add_argument("--seed-max", type=int, default=99999, help="Seed sampling upper bound (inclusive).")
    ap.add_argument("--run-root", type=Path, default=Path("data/simulation_output/thesis_final_runs_v1"), help="Target run root used in manifest out_dir.")
    ap.add_argument("--out-model", type=Path, default=Path("configs/thesis/final_model_v1.yaml"))
    ap.add_argument("--out-seeds", type=Path, default=Path("configs/thesis/final_seed_list_v1.json"))
    ap.add_argument("--out-manifest", type=Path, default=Path("configs/thesis/final_run_manifest_v1.csv"))
    ap.add_argument("--out-provenance", type=Path, default=Path("configs/thesis/freeze_provenance_v1.json"))
    args = ap.parse_args()

    source_root = Path(args.source_doe_root)
    spec_path = source_root / "doe_spec.json"
    points_path = source_root / "doe_design_points.csv"
    for p in (spec_path, points_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing source artifact: {p}")

    source_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    points = pd.read_csv(points_path)
    row = points[pd.to_numeric(points["design_id"], errors="coerce") == int(args.design_id)]
    if row.empty:
        raise ValueError(f"design_id={int(args.design_id)} not found in {points_path}")
    drow = row.iloc[0].to_dict()

    base_cfg = _load_yaml(Path(args.base_config))
    if "visualization" not in base_cfg or "output" not in base_cfg:
        raise ValueError(f"Base config must contain 'visualization' and 'output': {args.base_config}")

    model = dict(source_spec.get("frozen_model", {}))
    for k, v in drow.items():
        if str(k) == "design_id":
            continue
        if str(k) in INTEGER_MODEL_KEYS:
            model[str(k)] = int(round(float(v)))
        else:
            model[str(k)] = float(v)
    model["rule_idx"] = 2  # default; final manifest controls per-run rule.
    model["distance_idx"] = int(model.get("distance_idx", 0))
    model["seed"] = 0
    allowed_model_keys = set(ModelConfig.model_fields.keys())
    model = {k: v for k, v in model.items() if k in allowed_model_keys}

    simulation = dict(source_spec.get("frozen_simulation", {}))
    simulation["runs"] = 1
    simulation["base_seed"] = 0

    final_cfg: dict[str, Any] = {
        "model": model,
        "visualization": dict(base_cfg["visualization"]),
        "simulation": simulation,
        "output": dict(base_cfg["output"]),
    }

    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_model.write_text(yaml.safe_dump(final_cfg, sort_keys=False), encoding="utf-8")

    seeds_main = _build_seed_list(
        seed_count=int(args.seed_count_main),
        seed_rng=int(args.seed_rng),
        seed_min=int(args.seed_min),
        seed_max=int(args.seed_max),
    )
    approval_count = int(args.seed_count_approval)
    if approval_count > len(seeds_main):
        raise ValueError("seed-count-approval cannot exceed seed-count-main")

    out_seeds = Path(args.out_seeds)
    out_seeds.parent.mkdir(parents=True, exist_ok=True)
    seed_payload = {
        "version": "final_seed_list_v1",
        "source_doe_root": str(source_root),
        "selected_design_id": int(args.design_id),
        "seed_generation": {
            "method": "deterministic_random_sample_without_replacement",
            "rng_seed": int(args.seed_rng),
            "range_inclusive": [int(args.seed_min), int(args.seed_max)],
        },
        "main_rules": [0, 2, 3, 4, 5],
        "approval_rule": 1,
        "S_main": [int(s) for s in seeds_main],
        "S_approval": [int(s) for s in seeds_main[:approval_count]],
    }
    out_seeds.write_text(json.dumps(seed_payload, indent=2), encoding="utf-8")

    out_manifest = Path(args.out_manifest)
    _write_manifest(
        out_csv=out_manifest,
        run_root=Path(args.run_root),
        model_params=model,
        design_id=int(args.design_id),
        seeds_main=seeds_main,
        rules_main=[0, 2, 3, 4, 5],
        approval_context_count=approval_count,
    )

    out_prov = Path(args.out_provenance)
    out_prov.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    provenance = {
        "version": "freeze_provenance_v1",
        "generated_at_utc": now,
        "git_head": _git_head(),
        "git_dirty": bool(_git_dirty()),
        "source": {
            "doe_root": str(source_root),
            "doe_profile": str(source_spec.get("profile", "")),
            "selected_design_id": int(args.design_id),
        },
        "artifacts": {
            "final_model_v1_yaml": str(out_model),
            "final_seed_list_v1_json": str(out_seeds),
            "final_run_manifest_v1_csv": str(out_manifest),
            "sha256": {
                str(out_model): _sha256_file(out_model),
                str(out_seeds): _sha256_file(out_seeds),
                str(out_manifest): _sha256_file(out_manifest),
            },
        },
        "run_plan_counts": {
            "main_rules": 5,
            "main_seeds": len(seeds_main),
            "approval_seeds": approval_count,
            "expected_runs_total": int(5 * len(seeds_main) + approval_count),
        },
    }
    out_prov.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print(f"Wrote: {out_model}")
    print(f"Wrote: {out_seeds}")
    print(f"Wrote: {out_manifest}")
    print(f"Wrote: {out_prov}")
    print(f"Expected total runs: {provenance['run_plan_counts']['expected_runs_total']}")


if __name__ == "__main__":
    main()
