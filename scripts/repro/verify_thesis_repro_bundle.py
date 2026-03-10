from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import yaml

from src.analysis.doe_runner import rule_label


MANIFEST_FIELDNAMES = [
    "run_index",
    "design_id",
    "rule_idx",
    "rule_name",
    "seed",
    "family_role",
    "out_dir",
    "params_json",
    "params_hash",
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _infer_run_root_from_manifest(out_dir: str, design_id: int) -> str:
    s = str(out_dir).replace("\\", "/")
    marker = f"/design_{int(design_id):04d}/"
    idx = s.find(marker)
    if idx >= 0:
        return s[:idx]
    marker2 = f"design_{int(design_id):04d}/"
    idx2 = s.find(marker2)
    if idx2 >= 0:
        return s[:idx2].rstrip("/")
    return "data/simulation_output/thesis_final_runs_v1"


def _build_expected_manifest_rows(
    *,
    model_params: dict[str, Any],
    design_id: int,
    rules_main: list[int],
    seeds_main: list[int],
    approval_count: int,
    run_root: str,
) -> list[dict[str, Any]]:
    params_json = json.dumps(model_params, sort_keys=True, separators=(",", ":"))
    params_hash = hashlib.sha256(params_json.encode("utf-8")).hexdigest()

    rows: list[dict[str, Any]] = []
    for ridx in rules_main:
        rname = rule_label(int(ridx))
        for seed in seeds_main:
            out_dir = (
                f"{run_root}/design_{int(design_id):04d}/"
                f"rule_{rname}/seed_{int(seed):05d}/run_0"
            )
            rows.append(
                {
                    "run_index": int(len(rows)),
                    "design_id": int(design_id),
                    "rule_idx": int(ridx),
                    "rule_name": str(rname),
                    "seed": int(seed),
                    "family_role": "main",
                    "out_dir": out_dir,
                    "params_json": params_json,
                    "params_hash": params_hash,
                }
            )

    approval_rule_idx = 1
    approval_rule_name = rule_label(approval_rule_idx)
    for seed in seeds_main[: int(approval_count)]:
        out_dir = (
            f"{run_root}/design_{int(design_id):04d}/"
            f"rule_{approval_rule_name}/seed_{int(seed):05d}/run_0"
        )
        rows.append(
            {
                "run_index": int(len(rows)),
                "design_id": int(design_id),
                "rule_idx": int(approval_rule_idx),
                "rule_name": str(approval_rule_name),
                "seed": int(seed),
                "family_role": "approval_context",
                "out_dir": out_dir,
                "params_json": params_json,
                "params_hash": params_hash,
            }
        )

    return rows


def _manifest_rows_to_bytes(rows: list[dict[str, Any]]) -> bytes:
    sio = io.StringIO(newline="")
    writer = csv.DictWriter(sio, fieldnames=MANIFEST_FIELDNAMES)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return sio.getvalue().encode("utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify thesis reproducibility bundle artifacts and invariants.")
    ap.add_argument(
        "--freeze-provenance",
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
    args = ap.parse_args()

    freeze = _load_json(args.freeze_provenance)
    seed_payload = _load_json(args.final_seeds)
    config_payload = yaml.safe_load(args.final_model.read_text(encoding="utf-8"))
    if not isinstance(config_payload, dict):
        raise ValueError("final_model_v1.yaml must parse to an object.")
    model_params = config_payload.get("model")
    if not isinstance(model_params, dict):
        raise ValueError("final_model_v1.yaml is missing object key: model")

    # 1) Hash checks against published provenance block.
    artifacts = freeze.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("freeze provenance missing 'artifacts' object")
    hash_map = artifacts.get("sha256")
    if not isinstance(hash_map, dict):
        raise ValueError("freeze provenance missing 'artifacts.sha256' object")

    for rel, expected_hash in sorted(hash_map.items()):
        path = Path(str(rel))
        if not path.exists():
            raise FileNotFoundError(f"Published artifact missing: {path}")
        got = _sha256_file(path)
        if got != str(expected_hash):
            raise ValueError(f"Hash mismatch for {path}: expected={expected_hash} got={got}")

    # 2) Final manifest structural checks.
    with args.final_manifest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required_cols = set(MANIFEST_FIELDNAMES)
    if reader.fieldnames is None or set(reader.fieldnames) != required_cols:
        raise ValueError("final_run_manifest_v1.csv has unexpected columns")

    run_plan_counts = freeze.get("run_plan_counts", {})
    expected_total = int(run_plan_counts.get("expected_runs_total", 0))
    if len(rows) != expected_total:
        raise ValueError(f"Manifest row count mismatch: expected={expected_total} got={len(rows)}")

    run_indices = [int(row["run_index"]) for row in rows]
    if sorted(run_indices) != list(range(len(rows))):
        raise ValueError("Manifest run_index values are not a contiguous 0..N-1 sequence")

    main_rows = [row for row in rows if row.get("family_role") == "main"]
    approval_rows = [row for row in rows if row.get("family_role") == "approval_context"]
    expected_main = int(run_plan_counts.get("main_rules", 0)) * int(run_plan_counts.get("main_seeds", 0))
    expected_approval = int(run_plan_counts.get("approval_seeds", 0))
    if len(main_rows) != expected_main:
        raise ValueError(f"Main-family count mismatch: expected={expected_main} got={len(main_rows)}")
    if len(approval_rows) != expected_approval:
        raise ValueError(f"Approval-context count mismatch: expected={expected_approval} got={len(approval_rows)}")

    # 3) Determinism check: regenerate manifest bytes from frozen model+seed artifacts.
    design_id = int(seed_payload["selected_design_id"])
    rules_main = [int(x) for x in seed_payload["main_rules"]]
    seeds_main = [int(x) for x in seed_payload["S_main"]]
    s_approval = [int(x) for x in seed_payload["S_approval"]]
    if s_approval != seeds_main[: len(s_approval)]:
        raise ValueError("S_approval is not a prefix of S_main")

    if len(rows) == 0:
        raise ValueError("final_run_manifest_v1.csv is empty")
    run_root = _infer_run_root_from_manifest(rows[0]["out_dir"], design_id)

    expected_rows = _build_expected_manifest_rows(
        model_params=model_params,
        design_id=design_id,
        rules_main=rules_main,
        seeds_main=seeds_main,
        approval_count=len(s_approval),
        run_root=run_root,
    )
    expected_manifest_hash = hashlib.sha256(_manifest_rows_to_bytes(expected_rows)).hexdigest()
    published_manifest_hash = str(hash_map.get(str(args.final_manifest), ""))
    if expected_manifest_hash != published_manifest_hash:
        raise ValueError(
            "Manifest determinism check failed: regenerated hash does not match published hash. "
            f"expected={published_manifest_hash} regenerated={expected_manifest_hash}"
        )

    print("[OK] Hash block verified for published artifacts.")
    print(f"[OK] Final manifest row count verified: {len(rows)}")
    print(f"[OK] Family counts verified: main={len(main_rows)} approval_context={len(approval_rows)}")
    print(f"[OK] Deterministic manifest hash verified: {published_manifest_hash}")


if __name__ == "__main__":
    main()
