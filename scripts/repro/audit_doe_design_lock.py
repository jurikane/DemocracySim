from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.doe_scoring import score_designs


REQUIRED_BUNDLE_FILES = (
    "doe_spec.json",
    "doe_selection_spec.json",
    "doe_run_features.csv",
    "doe_design_scores.csv",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _assert_required_bundle_files(bundle_dir: Path) -> None:
    missing = [name for name in REQUIRED_BUNDLE_FILES if not (bundle_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Bundle missing required files: {sorted(missing)}")


def _float(x: Any, name: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for '{name}': {x!r}") from exc


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit DOE design-lock reproducibility from the lightweight published bundle.")
    ap.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("configs/thesis/doe_selection_bundle_v1"),
    )
    ap.add_argument(
        "--expected-design-id",
        type=int,
        default=None,
        help="Expected selected design. If omitted, read from configs/thesis/doe_selection_provenance_v1.json",
    )
    ap.add_argument(
        "--provenance",
        type=Path,
        default=Path("configs/thesis/doe_selection_provenance_v1.json"),
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=1e-12,
        help="Numerical tolerance for score_total comparison to published doe_design_scores.csv",
    )
    args = ap.parse_args()

    bundle_dir = Path(args.bundle_dir)
    _assert_required_bundle_files(bundle_dir)

    spec = _load_json(bundle_dir / "doe_spec.json")
    selection = _load_json(bundle_dir / "doe_selection_spec.json")
    run_features = pd.read_csv(bundle_dir / "doe_run_features.csv")
    published_scores = pd.read_csv(bundle_dir / "doe_design_scores.csv")

    expected_design_id = int(args.expected_design_id) if args.expected_design_id is not None else int(
        _load_json(args.provenance)["selected_design_id"]
    )

    thresholds = selection.get("thresholds", {})
    if not isinstance(thresholds, dict):
        raise ValueError("doe_selection_spec.json missing object key: thresholds")

    primary_rule_name = str(selection.get("primary_rule_name", spec.get("primary_rule_name", "approval")))

    required_primary_runs: int | None
    raw_required_runs = selection.get("required_primary_runs", None)
    if raw_required_runs is None:
        seeds = spec.get("seeds", [])
        required_primary_runs = int(len(seeds)) if isinstance(seeds, list) and len(seeds) > 0 else None
    else:
        required_primary_runs = int(raw_required_runs)

    recomputed = score_designs(
        run_features,
        primary_rule_name=primary_rule_name,
        weights=dict(selection.get("weights", {})),
        stage_weights=dict(selection.get("stage_weights", {})),
        required_primary_runs=required_primary_runs,
        min_winner_entropy_norm=_float(thresholds["min_winner_entropy_norm"], "min_winner_entropy_norm"),
        min_competitive_step_share=_float(thresholds["min_competitive_step_share"], "min_competitive_step_share"),
        puzzle_dominance_share_score_low=_float(
            thresholds["puzzle_dominance_share_score_low"],
            "puzzle_dominance_share_score_low",
        ),
        puzzle_dominance_share_score_high=_float(
            thresholds["puzzle_dominance_share_score_high"],
            "puzzle_dominance_share_score_high",
        ),
        turnout_start_score_low=_float(thresholds["turnout_start_score_low"], "turnout_start_score_low"),
        turnout_start_score_high=_float(thresholds["turnout_start_score_high"], "turnout_start_score_high"),
        turnout_end_score_low=_float(thresholds["turnout_end_score_low"], "turnout_end_score_low"),
        turnout_end_score_high=_float(thresholds["turnout_end_score_high"], "turnout_end_score_high"),
        turnout_drop_score_good_max=_float(
            thresholds["turnout_drop_score_good_max"],
            "turnout_drop_score_good_max",
        ),
        turnout_drop_score_zero_at=_float(thresholds["turnout_drop_score_zero_at"], "turnout_drop_score_zero_at"),
        turnout_decline_score_good_max=_float(
            thresholds["turnout_decline_score_good_max"],
            "turnout_decline_score_good_max",
        ),
        turnout_decline_score_zero_at=_float(
            thresholds["turnout_decline_score_zero_at"],
            "turnout_decline_score_zero_at",
        ),
        turnout_outside_band_share_good_max=_float(
            thresholds["turnout_outside_band_share_good_max"],
            "turnout_outside_band_share_good_max",
        ),
        turnout_outside_band_share_zero_at=_float(
            thresholds["turnout_outside_band_share_zero_at"],
            "turnout_outside_band_share_zero_at",
        ),
        quality_component_weights=selection.get("quality_component_weights"),
    )

    if len(recomputed) == 0:
        raise ValueError("Recomputed ranking is empty.")

    got_top = int(recomputed.iloc[0]["design_id"])
    if got_top != expected_design_id:
        raise ValueError(
            f"Design-lock audit failed: expected top design_id={expected_design_id}, got={got_top}."
        )

    # Compare published and recomputed scores for reproducibility confidence.
    merged = recomputed[["design_id", "score_total"]].merge(
        published_scores[["design_id", "score_total"]],
        on="design_id",
        how="inner",
        suffixes=("_recomputed", "_published"),
    )
    if len(merged) == 0:
        raise ValueError("No overlap between recomputed and published design scores.")

    max_abs_diff = float(
        np.max(np.abs(merged["score_total_recomputed"].to_numpy() - merged["score_total_published"].to_numpy()))
    )
    if max_abs_diff > float(args.tol):
        raise ValueError(
            "Score reproducibility mismatch: max |recomputed-published| exceeds tolerance. "
            f"tol={args.tol} max_abs_diff={max_abs_diff}"
        )

    print(f"[OK] Recomputed top design matches expected lock: design_id={expected_design_id}")
    print(f"[OK] Max |score_total_recomputed - score_total_published| = {max_abs_diff:.3e}")
    print("[OK] DOE lock audit passed.")


if __name__ == "__main__":
    main()
