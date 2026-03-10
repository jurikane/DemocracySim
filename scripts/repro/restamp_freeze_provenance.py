from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


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


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Restamp freeze provenance with full reproducibility hash block.")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("configs/thesis/freeze_provenance_v1.json"),
    )
    ap.add_argument(
        "--final-model",
        type=Path,
        default=Path("configs/thesis/final_model_v1.yaml"),
    )
    ap.add_argument(
        "--final-seeds",
        type=Path,
        default=Path("configs/thesis/final_seed_list_v1.json"),
    )
    ap.add_argument(
        "--final-manifest",
        type=Path,
        default=Path("configs/thesis/final_run_manifest_v1.csv"),
    )
    ap.add_argument(
        "--doe-selection-provenance",
        type=Path,
        default=Path("configs/thesis/doe_selection_provenance_v1.json"),
    )
    ap.add_argument(
        "--doe-bundle-dir",
        type=Path,
        default=Path("configs/thesis/doe_selection_bundle_v1"),
    )
    ap.add_argument(
        "--release-tag",
        type=str,
        default="",
        help="Optional immutable release tag (for example thesis-freeze-v1).",
    )
    ap.add_argument(
        "--release-url",
        type=str,
        default="",
        help="Optional release URL.",
    )
    args = ap.parse_args()

    for p in [
        args.final_model,
        args.final_seeds,
        args.final_manifest,
        args.doe_selection_provenance,
        args.doe_bundle_dir,
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    seed_payload = _load_json(args.final_seeds)
    doe_sel = _load_json(args.doe_selection_provenance)

    bundle_files = sorted(
        [p for p in args.doe_bundle_dir.glob("*") if p.is_file()]
    )
    if not bundle_files:
        raise FileNotFoundError(f"No files found in DOE bundle dir: {args.doe_bundle_dir}")

    hash_block: dict[str, str] = {}
    core_artifacts = [
        args.final_model,
        args.final_seeds,
        args.final_manifest,
        args.doe_selection_provenance,
    ]
    for p in core_artifacts + bundle_files:
        hash_block[str(p)] = _sha256_file(p)

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    s_main = seed_payload.get("S_main", [])
    main_rules = seed_payload.get("main_rules", [])
    s_approval = seed_payload.get("S_approval", [])

    provenance = {
        "version": "freeze_provenance_v1",
        "generated_at_utc": now,
        "git_head": _git_head(),
        "git_dirty": bool(_git_dirty()),
        "release": {
            "tag": (str(args.release_tag) if str(args.release_tag) else None),
            "url": (str(args.release_url) if str(args.release_url) else None),
        },
        "source": {
            "doe_batch_id": str(doe_sel.get("doe_batch_id", "DOE-UNKNOWN")),
            "doe_profile": str(doe_sel.get("doe_profile", "")),
            "selected_design_id": int(doe_sel.get("selected_design_id", seed_payload.get("selected_design_id", -1))),
            "doe_selection_provenance_v1_json": str(args.doe_selection_provenance),
            "doe_selection_bundle_dir": str(args.doe_bundle_dir),
            "doe_selection_bundle_files": [str(p) for p in bundle_files],
        },
        "artifacts": {
            "final_model_v1_yaml": str(args.final_model),
            "final_seed_list_v1_json": str(args.final_seeds),
            "final_run_manifest_v1_csv": str(args.final_manifest),
            "doe_selection_provenance_v1_json": str(args.doe_selection_provenance),
            "doe_selection_bundle_dir": str(args.doe_bundle_dir),
            "sha256": hash_block,
        },
        "run_plan_counts": {
            "main_rules": int(len(main_rules)),
            "main_seeds": int(len(s_main)),
            "approval_seeds": int(len(s_approval)),
            "expected_runs_total": int(len(main_rules) * len(s_main) + len(s_approval)),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
