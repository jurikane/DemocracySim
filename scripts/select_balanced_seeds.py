from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.analysis.doe_runner import (
    apply_doe_overrides,
    get_doe_profile,
    midpoint_params_from_ranges,
    select_farthest_seeds_from_descriptors,
)
from src.config.loader import load_config
from src.model_setup import make_model


@dataclass(frozen=True)
class SeedProbe:
    seed: int
    ratio_max_to_min: float
    personality_group_distribution: list[float]
    descriptor: np.ndarray


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Select a balanced seed set by filtering candidate seeds on "
            "personality-group imbalance, then maximizing descriptor spread."
        )
    )
    p.add_argument("--config", "-c", default="doe.yaml", help="Config path or name under configs/.")
    p.add_argument("--doe-profile", required=True, help="DOE profile name.")
    p.add_argument("--target", type=int, required=True, help="Number of seeds to select.")
    p.add_argument("--rule-idx", type=int, default=1, help="Rule index used during probing.")
    p.add_argument(
        "--candidate-start",
        type=int,
        default=100,
        help="Start value for candidate seed range (inclusive).",
    )
    p.add_argument(
        "--candidate-count",
        type=int,
        default=2000,
        help="Number of candidate seeds to probe.",
    )
    p.add_argument(
        "--max-imbalance-ratio",
        type=float,
        default=3.0,
        help="Keep only seeds with max(group_share)/min(group_share) <= this value.",
    )
    p.add_argument(
        "--out-json",
        type=str,
        default="",
        help="Optional output path for a JSON summary (selected seeds + diagnostics).",
    )
    return p.parse_args()


def _ratio_max_to_min(pg: np.ndarray) -> float:
    pg = np.asarray(pg, dtype=float).reshape(-1)
    positive = pg[pg > 0]
    if positive.size == 0:
        return float("inf")
    return float(np.max(positive) / np.min(positive))


def _probe_seed(
    *,
    cfg,
    seed: int,
    rule_idx: int,
    params: dict[str, float],
    frozen_model: dict[str, Any] | None,
    frozen_simulation: dict[str, Any] | None,
) -> SeedProbe:
    cfg_probe = apply_doe_overrides(
        cfg,
        params=params,
        rule_idx=int(rule_idx),
        base_seed=int(seed),
        frozen_model=frozen_model,
        frozen_simulation=frozen_simulation,
    )
    cfg_probe.model.seed = int(seed)
    model = make_model(cfg_probe.model, enable_datacollector=False)
    g = np.asarray(model.global_color_dst, dtype=float).reshape(-1)
    pg = np.asarray(model.personality_group_distribution, dtype=float).reshape(-1)
    desc = np.concatenate([g, pg]).astype(float)
    ratio = _ratio_max_to_min(pg)
    return SeedProbe(
        seed=int(seed),
        ratio_max_to_min=float(ratio),
        personality_group_distribution=[float(x) for x in pg.tolist()],
        descriptor=desc,
    )


def main() -> None:
    args = _parse_args()
    if args.target <= 0:
        raise ValueError("--target must be >= 1")
    if args.candidate_count <= 0:
        raise ValueError("--candidate-count must be >= 1")
    if args.max_imbalance_ratio <= 1.0:
        raise ValueError("--max-imbalance-ratio must be > 1.0")

    cfg = load_config(args.config)
    profile = get_doe_profile(args.doe_profile)
    ranges = profile.get("ranges")
    if not isinstance(ranges, dict):
        raise ValueError(f"DOE profile {args.doe_profile!r} has no ranges.")
    frozen_model = profile.get("frozen_model")
    frozen_simulation = profile.get("frozen_simulation")
    params = midpoint_params_from_ranges(ranges)

    seeds = list(range(int(args.candidate_start), int(args.candidate_start) + int(args.candidate_count)))

    probed: list[SeedProbe] = []
    for s in seeds:
        probed.append(
            _probe_seed(
                cfg=cfg,
                seed=s,
                rule_idx=int(args.rule_idx),
                params=params,
                frozen_model=frozen_model,
                frozen_simulation=frozen_simulation,
            )
        )

    ratios = np.asarray([p.ratio_max_to_min for p in probed], dtype=float)
    filtered = [p for p in probed if p.ratio_max_to_min <= float(args.max_imbalance_ratio)]
    if len(filtered) < int(args.target):
        raise RuntimeError(
            "Not enough balanced candidates. "
            f"Need target={args.target}, got={len(filtered)} after filtering "
            f"(threshold={args.max_imbalance_ratio}). Increase --candidate-count "
            "or relax --max-imbalance-ratio."
        )

    filtered_desc = {int(p.seed): np.asarray(p.descriptor, dtype=float) for p in filtered}
    selected = select_farthest_seeds_from_descriptors(
        filtered_desc,
        target_count=int(args.target),
    )

    selected_set = set(int(s) for s in selected)
    selected_ratios = np.asarray(
        [p.ratio_max_to_min for p in filtered if int(p.seed) in selected_set],
        dtype=float,
    )

    summary = {
        "config": str(args.config),
        "doe_profile": str(args.doe_profile),
        "rule_idx": int(args.rule_idx),
        "candidate_start": int(args.candidate_start),
        "candidate_count": int(args.candidate_count),
        "max_imbalance_ratio": float(args.max_imbalance_ratio),
        "target": int(args.target),
        "candidate_ratio_quantiles": {
            "p10": float(np.quantile(ratios, 0.10)),
            "p25": float(np.quantile(ratios, 0.25)),
            "p50": float(np.quantile(ratios, 0.50)),
            "p75": float(np.quantile(ratios, 0.75)),
            "p90": float(np.quantile(ratios, 0.90)),
        },
        "filtered_count": int(len(filtered)),
        "selected_ratio_quantiles": {
            "p10": float(np.quantile(selected_ratios, 0.10)),
            "p25": float(np.quantile(selected_ratios, 0.25)),
            "p50": float(np.quantile(selected_ratios, 0.50)),
            "p75": float(np.quantile(selected_ratios, 0.75)),
            "p90": float(np.quantile(selected_ratios, 0.90)),
            "max": float(np.max(selected_ratios)),
        },
        "selected_seeds": [int(s) for s in selected],
    }

    print("Selected seeds (comma-separated):")
    print(",".join(str(int(s)) for s in selected))
    print("\nSelection summary:")
    print(json.dumps(summary, indent=2))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote summary JSON: {out_path}")


if __name__ == "__main__":
    main()
