from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import json
import numpy as np
import hashlib


RULE_LABELS = {
    0: "majority",
    1: "approval",
    2: "utilitarian",
    3: "borda",
}


# Phase-1 DOE ranges (confirmed)
DEFAULT_DOE_RANGES: dict[str, tuple[float, float]] = {
    "election_cost_rate": (0.001, 0.10),
    "reward_rate_common": (0.00, 0.12),
    "reward_rate_personal": (0.00, 0.30),
    "break_even_distance_common": (0.15, 0.45),
    "break_even_distance_personal": (0.45, 0.85),
    "election_impact_on_mutation": (1.0, 3.0),
    "mu": (0.15, 1.00),
    "participation_alpha": (0.01, 0.20),
    "participation_beta": (2.5, 9.5),
    "participation_init_q": (0.15, 1.2),
    "altruism_static": (0.25, 0.75),
}


# Frozen model settings for DOE phase-1
DEFAULT_FROZEN_MODEL: dict[str, Any] = {
    "distance_idx": 0,
    "abstention_share": 1.0,
    "participation_q_max": 2.0,
    "bias_toward_participation": 0.0,
    "altruism_learning": False,
    "altruism_alpha": 0.05,
    "altruism_init": 0.5,
    "altruism_clip_min": 0.0,
    "altruism_clip_max": 1.0,
    "satisfaction_mode": "area",
    "satisfaction_baseline_alpha": 0.1,
    "participation_baseline_alpha": 0.1,
    "initial_agent_assets": 100.0,
    "heterogeneity": 0.3,
    "known_cells": 10,
    "personal_preference_peakedness": 1.0,
    "num_agents": 120,
    "num_colors": 5,
    "num_personality_groups": 4,
    "height": 30,
    "width": 50,
    "num_areas": 1,
    "av_area_height": 30,
    "av_area_width": 50,
    "area_size_variance": 0.0,
    "color_patches_steps": 0,
    "patch_power": 1.0,
}


DEFAULT_FROZEN_SIM: dict[str, Any] = {
    "runs": 1,
    "num_steps": 150,
    "store_grid": False,
    "grid_interval": 1,
}


@dataclass(frozen=True)
class DOERunTask:
    design_id: int
    seed: int
    rule_idx: int
    out_dir: Path
    params: dict[str, float]


def sample_design_points(
    *,
    num_points: int,
    rng,
    ranges: dict[str, tuple[float, float]] | None = None,
    max_tries_per_point: int = 2000,
) -> list[dict[str, float]]:
    """Sample DOE design points by uniform draws with algebra safety filtering."""
    if num_points <= 0:
        raise ValueError("num_points must be >= 1")
    r = DEFAULT_DOE_RANGES if ranges is None else ranges
    points: list[dict[str, float]] = []
    keys = sorted(r.keys())
    for _ in range(num_points):
        accepted = False
        for _try in range(max_tries_per_point):
            p = {k: float(rng.uniform(r[k][0], r[k][1])) for k in keys}
            if _passes_rate_sum_constraint(p):
                points.append(p)
                accepted = True
                break
        if not accepted:
            raise RuntimeError("Could not sample a valid DOE point under constraints.")
    return points


def _passes_rate_sum_constraint(p: dict[str, float]) -> bool:
    return (
        float(p["election_cost_rate"])
        + float(p["reward_rate_common"])
        + float(p["reward_rate_personal"])
        <= 0.9
    )


def midpoint_params_from_ranges(ranges: dict[str, tuple[float, float]]) -> dict[str, float]:
    """Return midpoint parameter set for probe runs."""
    out: dict[str, float] = {}
    for k, (lo, hi) in ranges.items():
        out[k] = float((float(lo) + float(hi)) / 2.0)
    return out


def select_farthest_seeds_from_descriptors(
    descriptors: dict[int, np.ndarray],
    *,
    target_count: int,
) -> list[int]:
    """Greedy max-min farthest-point seed selection in standardized descriptor space."""
    if target_count <= 0:
        raise ValueError("target_count must be >= 1")
    if not descriptors:
        raise ValueError("descriptors must not be empty")
    seeds = sorted(int(s) for s in descriptors.keys())
    if target_count > len(seeds):
        raise ValueError(f"target_count={target_count} exceeds candidate seeds={len(seeds)}")

    vecs = [np.asarray(descriptors[s], dtype=float).reshape(-1) for s in seeds]
    dim = int(vecs[0].size)
    if any(int(v.size) != dim for v in vecs):
        raise ValueError("All descriptor vectors must have same dimensionality.")
    x = np.vstack(vecs)

    mu = np.nanmean(x, axis=0)
    sigma = np.nanstd(x, axis=0)
    sigma[sigma < 1e-12] = 1.0
    xz = (x - mu) / sigma

    centroid = np.mean(xz, axis=0)
    d0 = np.linalg.norm(xz - centroid, axis=1)
    first_idx = int(np.argmax(d0))
    selected_idx: list[int] = [first_idx]
    remaining: set[int] = set(range(len(seeds)))
    remaining.remove(first_idx)

    while len(selected_idx) < target_count:
        best_i = None
        best_val = -1.0
        for i in sorted(remaining):
            dist_to_sel = [float(np.linalg.norm(xz[i] - xz[j])) for j in selected_idx]
            score = float(min(dist_to_sel)) if dist_to_sel else 0.0
            if score > best_val + 1e-12:
                best_val = score
                best_i = i
        assert best_i is not None
        selected_idx.append(best_i)
        remaining.remove(best_i)

    return [seeds[i] for i in selected_idx]


def select_stratified_seeds(
    cfg,
    *,
    target_count: int,
    candidate_seeds: list[int],
    probe_rule_idx: int = 1,
    probe_params: dict[str, float] | None = None,
) -> list[int]:
    """Select spread-out seeds based on initial-state descriptors from model instantiation."""
    uniq = sorted(set(int(s) for s in candidate_seeds))
    if target_count > len(uniq):
        raise ValueError(f"target_count={target_count} exceeds candidate seeds={len(uniq)}")
    params = midpoint_params_from_ranges(DEFAULT_DOE_RANGES) if probe_params is None else probe_params

    # Local import to keep DOE utility module lightweight for non-selection paths.
    from src.model_setup import make_model

    descriptors: dict[int, np.ndarray] = {}
    for seed in uniq:
        cfg_probe = apply_doe_overrides(cfg, params=params, rule_idx=int(probe_rule_idx), base_seed=int(seed))
        cfg_probe.model.seed = int(seed)
        model = make_model(cfg_probe.model, enable_datacollector=False)
        g = np.asarray(model.global_color_dst, dtype=float).reshape(-1)
        pg = np.asarray(model.personality_group_distribution, dtype=float).reshape(-1)
        desc = np.concatenate([g, pg]).astype(float)
        descriptors[int(seed)] = desc
    return select_farthest_seeds_from_descriptors(descriptors, target_count=target_count)


def write_seed_selection_manifest(
    *,
    out_root: Path,
    mode: str,
    selected_seeds: list[int],
    candidate_seeds: list[int] | None = None,
    probe_rule_idx: int | None = None,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "seed_mode": str(mode),
        "selected_seeds": [int(s) for s in selected_seeds],
    }
    if candidate_seeds is not None:
        payload["candidate_seeds"] = [int(s) for s in candidate_seeds]
    if probe_rule_idx is not None:
        payload["probe_rule_idx"] = int(probe_rule_idx)
    (out_root / "doe_seed_selection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_run_plan(
    *,
    out_root: Path,
    design_points: list[dict[str, float]],
    seeds: list[int],
    primary_rule_idx: int = 1,
    robust_rule_idx: int = 2,
    include_robustness: bool = True,
    robust_every: int = 1,
) -> list[DOERunTask]:
    if robust_every <= 0:
        raise ValueError("robust_every must be >= 1")
    tasks: list[DOERunTask] = []
    for i, params in enumerate(design_points):
        design_dir = out_root / f"design_{i:04d}"
        for seed in seeds:
            pri_out = design_dir / f"rule_{rule_label(primary_rule_idx)}" / f"seed_{seed:05d}" / "run_0"
            tasks.append(
                DOERunTask(
                    design_id=i,
                    seed=int(seed),
                    rule_idx=int(primary_rule_idx),
                    out_dir=pri_out,
                    params=params,
                )
            )
            if include_robustness and (i % robust_every == 0):
                rob_out = design_dir / f"rule_{rule_label(robust_rule_idx)}" / f"seed_{seed:05d}" / "run_0"
                tasks.append(
                    DOERunTask(
                        design_id=i,
                        seed=int(seed),
                        rule_idx=int(robust_rule_idx),
                        out_dir=rob_out,
                        params=params,
                    )
                )
    return tasks


def apply_doe_overrides(cfg, *, params: dict[str, float], rule_idx: int, base_seed: int):
    """Return a deep-copied AppConfig with DOE overrides applied."""
    c = cfg.model_copy(deep=True)
    for k, v in DEFAULT_FROZEN_MODEL.items():
        setattr(c.model, k, v)
    for k, v in params.items():
        setattr(c.model, k, float(v))
    c.model.rule_idx = int(rule_idx)
    for k, v in DEFAULT_FROZEN_SIM.items():
        setattr(c.simulation, k, v)
    c.simulation.base_seed = int(base_seed)
    return c


def rule_label(rule_idx: int) -> str:
    return RULE_LABELS.get(int(rule_idx), f"rule_{int(rule_idx)}")


def write_design_manifest(
    *,
    out_root: Path,
    design_points: list[dict[str, float]],
    seeds: list[int],
    primary_rule_idx: int,
    robust_rule_idx: int,
    include_robustness: bool,
    robust_every: int,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    spec = {
        "design_points": len(design_points),
        "seeds": [int(s) for s in seeds],
        "primary_rule_idx": int(primary_rule_idx),
        "primary_rule_name": rule_label(primary_rule_idx),
        "robust_rule_idx": int(robust_rule_idx),
        "robust_rule_name": rule_label(robust_rule_idx),
        "include_robustness": bool(include_robustness),
        "robust_every": int(robust_every),
        "ranges": {k: [float(v[0]), float(v[1])] for k, v in DEFAULT_DOE_RANGES.items()},
        "frozen_model": dict(DEFAULT_FROZEN_MODEL),
        "frozen_simulation": dict(DEFAULT_FROZEN_SIM),
        "constraint": "election_cost_rate + reward_rate_common + reward_rate_personal <= 0.9",
    }
    (out_root / "doe_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    points_path = out_root / "doe_design_points.csv"
    keys = sorted(DEFAULT_DOE_RANGES.keys())
    with points_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["design_id", *keys])
        w.writeheader()
        for i, p in enumerate(design_points):
            row = {"design_id": i}
            for k in keys:
                row[k] = float(p[k])
            w.writerow(row)


def write_run_manifest(*, out_root: Path, plan: list[DOERunTask]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "doe_run_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run_index",
                "design_id",
                "seed",
                "rule_idx",
                "rule_name",
                "out_dir",
                "params_json",
                "params_hash",
            ],
        )
        w.writeheader()
        for i, task in enumerate(plan):
            params_json = json.dumps(task.params, sort_keys=True, separators=(",", ":"))
            params_hash = hashlib.sha256(params_json.encode("utf-8")).hexdigest()
            w.writerow(
                {
                    "run_index": int(i),
                    "design_id": int(task.design_id),
                    "seed": int(task.seed),
                    "rule_idx": int(task.rule_idx),
                    "rule_name": rule_label(int(task.rule_idx)),
                    "out_dir": str(task.out_dir),
                    "params_json": params_json,
                    "params_hash": params_hash,
                }
            )
