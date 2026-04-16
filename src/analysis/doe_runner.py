from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import json
import numpy as np
import hashlib
import copy


RULE_LABELS = {
    0: "plurality",
    1: "approval",
    2: "utilitarian",
    3: "borda",
    4: "schulze",
    5: "random",
}

INTEGER_DOE_KEYS: set[str] = {"known_cells"}


def _known_cells_probe_bounds(model_cfg: dict[str, Any]) -> tuple[float, float]:
    num_colors = int(model_cfg["num_colors"])
    grid_fields = int(model_cfg["height"]) * int(model_cfg["width"])
    max_known = max(num_colors, int(np.floor(0.02 * float(grid_fields))))
    return float(num_colors), float(max_known)


DEFAULT_DOE_RANGES: dict[str, tuple[float, float]] = {
    "election_cost_rate": (0.001, 0.10),
    "reward_rate_personal": (0.00, 0.30),
    "break_even_distance_common": (0.15, 0.45),
    "election_impact_on_mutation": (1.0, 3.0),
    "mu": (0.15, 1.00),
    "participation_alpha": (0.01, 0.20),
    "participation_beta": (2.5, 9.5),
    "participation_init_q": (0.15, 1.2),
    "altruism_static": (0.25, 0.75),
}


DEFAULT_FROZEN_MODEL: dict[str, Any] = {
    "distance_idx": 0,
    "participation_q_max": 2.0,
    "altruism_mode": "satisfaction",
    "altruism_response_gamma": 1.0,
    "altruism_learning": False,
    "altruism_alpha": 0.05,
    "altruism_init": 0.5,
    "altruism_clip_min": 0.0,
    "altruism_clip_max": 1.0,
    "satisfaction_mode": "area",
    "satisfaction_baseline_alpha": 0.1,
    "quality_target_mode": "puzzle",
    "puzzle_local_kappa": 30.0,
    "puzzle_shock_prob": 0.05,
    "participation_baseline_alpha": 0.1,
    "initial_agent_assets": 100.0,
    "heterogeneity": 0.3,
    "known_cells": 10,
    "personal_preference_peakedness": 1.0,
    "num_agents": 100,
    "num_colors": 4,
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
    "num_steps": 250,
    "store_grid": False,
    "grid_interval": 1,
}


DEFAULT_DOE_PROFILES: dict[str, dict[str, Any]] = {
    "phase1": {
        "name": "phase1",
        "ranges": dict(DEFAULT_DOE_RANGES),
        "frozen_model": dict(DEFAULT_FROZEN_MODEL),
        "frozen_simulation": dict(DEFAULT_FROZEN_SIM),
    },
    "phase2_altruism_learning": {
        "name": "phase2_altruism_learning",
        "ranges": {
            "election_cost_rate": (0.001, 0.10),
            "reward_rate_personal": (0.00, 0.30),
            "break_even_distance_common": (0.15, 0.45),
            "election_impact_on_mutation": (1.0, 3.0),
            "mu": (0.15, 1.00),
            "participation_alpha": (0.01, 0.20),
            "participation_beta": (2.5, 9.5),
            "participation_init_q": (0.15, 1.2),
            "altruism_alpha": (0.01, 0.08),
            "altruism_init": (0.20, 0.80),
        },
        "frozen_model": {
            **DEFAULT_FROZEN_MODEL,
            "altruism_mode": "surprise_learning",
            "altruism_learning": True,
        },
        "frozen_simulation": dict(DEFAULT_FROZEN_SIM),
    },
    "phase2_altruism_probe": {
        "name": "phase2_altruism_probe",
        "ranges": {
            "election_cost_rate": (0.001, 0.10),
            "reward_rate_personal": (0.00, 0.30),
            "break_even_distance_common": (0.15, 0.45),
            "election_impact_on_mutation": (1.0, 3.0),
            "mu": (0.15, 1.00),
            "participation_alpha": (0.01, 0.20),
            "participation_beta": (2.5, 9.5),
            "participation_init_q": (0.15, 1.2),
            "altruism_alpha": (0.01, 0.08),
            "altruism_init": (0.20, 0.80),
            "satisfaction_baseline_alpha": (0.02, 0.25),
            "known_cells": _known_cells_probe_bounds(DEFAULT_FROZEN_MODEL),
        },
        "frozen_model": {
            **DEFAULT_FROZEN_MODEL,
            "altruism_mode": "surprise_learning",
            "altruism_learning": True,
        },
        "frozen_simulation": dict(DEFAULT_FROZEN_SIM),
    },
    "phase3_puzzle_main": {
        "name": "phase3_puzzle_main",
        "ranges": {
            "election_cost_rate": (0.001, 0.05),
            "reward_rate_personal": (0.03, 0.25),
            "break_even_distance_common": (0.25, 0.60),
            "election_impact_on_mutation": (0.75, 3.0),
            "mu": (0.05, 0.50),
            "participation_alpha": (0.03, 0.20),
            "participation_beta": (2.5, 10.0),
            "participation_init_q": (0.01, 1.20),
            "known_cells": _known_cells_probe_bounds(DEFAULT_FROZEN_MODEL),
            "altruism_response_gamma": (0.9, 1.00),
            "puzzle_local_kappa": (5.0, 80.0),
            "puzzle_shock_prob": (0.00, 0.15),
        },
        "frozen_model": {
            **DEFAULT_FROZEN_MODEL,
            "altruism_mode": "satisfaction",
            "altruism_learning": False,
            "participation_signal_mode": "group_centered_delta_rel_plus_fee",
            "participation_signal_fee_weight": 1.0,
            "participation_signal_group_shrink_k": 10.0,
            "participation_signal_clip": 0.25,
        },
        "frozen_simulation": dict(DEFAULT_FROZEN_SIM),
    },
}


def list_doe_profiles() -> tuple[str, ...]:
    return tuple(DEFAULT_DOE_PROFILES.keys())


def get_doe_profile(name: str) -> dict[str, Any]:
    key = str(name).strip()
    if key not in DEFAULT_DOE_PROFILES:
        raise ValueError(
            f"Unknown DOE profile: {name!r}. Available: {sorted(DEFAULT_DOE_PROFILES.keys())}"
        )
    return copy.deepcopy(DEFAULT_DOE_PROFILES[key])


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
            p: dict[str, float] = {}
            for k in keys:
                lo = float(r[k][0])
                hi = float(r[k][1])
                if k in INTEGER_DOE_KEYS:
                    p[k] = float(int(rng.integers(int(lo), int(hi) + 1)))
                else:
                    p[k] = float(rng.uniform(lo, hi))
            if _passes_rate_sum_constraint(p):
                points.append(p)
                accepted = True
                break
        if not accepted:
            raise RuntimeError("Could not sample a valid DOE point under constraints.")
    return points


def _passes_rate_sum_constraint(p: dict[str, float]) -> bool:
    if "election_cost_rate" not in p or "reward_rate_personal" not in p:
        return True
    return float(p["election_cost_rate"]) + float(p["reward_rate_personal"]) <= 0.9


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
    ranges: dict[str, tuple[float, float]] | None = None,
    frozen_model: dict[str, Any] | None = None,
    frozen_simulation: dict[str, Any] | None = None,
) -> list[int]:
    """Select spread-out seeds based on initial-state descriptors from model instantiation."""
    uniq = sorted(set(int(s) for s in candidate_seeds))
    if target_count > len(uniq):
        raise ValueError(f"target_count={target_count} exceeds candidate seeds={len(uniq)}")
    r = DEFAULT_DOE_RANGES if ranges is None else ranges
    params = midpoint_params_from_ranges(r) if probe_params is None else probe_params

    from src.model_setup import make_model

    descriptors: dict[int, np.ndarray] = {}
    for seed in uniq:
        cfg_probe = apply_doe_overrides(
            cfg,
            params=params,
            rule_idx=int(probe_rule_idx),
            base_seed=int(seed),
            frozen_model=frozen_model,
            frozen_simulation=frozen_simulation,
        )
        cfg_probe.model.seed = int(seed)
        model = make_model(cfg_probe.model, enable_datacollector=False)
        g = np.asarray(model.global_color_dst, dtype=float).reshape(-1)
        pg = np.asarray(model.personality_group_distribution, dtype=float).reshape(-1)
        descriptors[int(seed)] = np.concatenate([g, pg]).astype(float)
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


def apply_doe_overrides(
    cfg,
    *,
    params: dict[str, float],
    rule_idx: int,
    base_seed: int,
    frozen_model: dict[str, Any] | None = None,
    frozen_simulation: dict[str, Any] | None = None,
):
    """Return a deep-copied AppConfig with DOE overrides applied."""
    c = cfg.model_copy(deep=True)
    fm = DEFAULT_FROZEN_MODEL if frozen_model is None else frozen_model
    fs = DEFAULT_FROZEN_SIM if frozen_simulation is None else frozen_simulation
    for k, v in fm.items():
        setattr(c.model, k, v)
    for k, v in params.items():
        if k in INTEGER_DOE_KEYS:
            setattr(c.model, k, int(round(float(v))))
        else:
            setattr(c.model, k, float(v))
    c.model.rule_idx = int(rule_idx)
    for k, v in fs.items():
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
    ranges: dict[str, tuple[float, float]] | None = None,
    frozen_model: dict[str, Any] | None = None,
    frozen_simulation: dict[str, Any] | None = None,
    profile_name: str = "phase1",
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    r = DEFAULT_DOE_RANGES if ranges is None else ranges
    fm = DEFAULT_FROZEN_MODEL if frozen_model is None else frozen_model
    fs = DEFAULT_FROZEN_SIM if frozen_simulation is None else frozen_simulation
    spec = {
        "profile": str(profile_name),
        "design_points": len(design_points),
        "seeds": [int(s) for s in seeds],
        "primary_rule_idx": int(primary_rule_idx),
        "primary_rule_name": rule_label(primary_rule_idx),
        "robust_rule_idx": int(robust_rule_idx),
        "robust_rule_name": rule_label(robust_rule_idx),
        "include_robustness": bool(include_robustness),
        "robust_every": int(robust_every),
        "ranges": {k: [float(v[0]), float(v[1])] for k, v in r.items()},
        "frozen_model": dict(fm),
        "frozen_simulation": dict(fs),
        "constraint": "election_cost_rate + reward_rate_personal <= 0.9",
    }
    (out_root / "doe_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    points_path = out_root / "doe_design_points.csv"
    keys = sorted(r.keys())
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
