from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable

import numpy as np
import pandas as pd

from src.analysis.doe_scoring import (
    PRIMARY_QUALITY_COMPONENT_KEYS,
    apply_hard_gates,
    load_selection_objective,
    score_designs,
)


def _resolve_default_doe_root() -> Path | None:
    base = Path("data") / "simulation_output"
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir() and (p / "doe_run_features.csv").exists()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _prompt_text(label: str, default: str, help_text: str) -> str:
    while True:
        val = input(f"{label} [{default}] - {help_text}: ").strip()
        if val == "":
            return default
        return val


def _prompt_int(label: str, default: int, help_text: str, *, min_value: int | None = None) -> int:
    while True:
        raw = input(f"{label} [{default}] - {help_text}: ").strip()
        if raw == "":
            return int(default)
        try:
            v = int(raw)
        except Exception:
            print("Enter a valid integer.")
            continue
        if min_value is not None and v < min_value:
            print(f"Must be >= {min_value}.")
            continue
        return v


def _prompt_float(
    label: str,
    default: float,
    help_text: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    while True:
        raw = input(f"{label} [{default}] - {help_text}: ").strip()
        if raw == "":
            return float(default)
        try:
            v = float(raw)
        except Exception:
            print("Enter a valid number.")
            continue
        if min_value is not None and v < min_value:
            print(f"Must be >= {min_value}.")
            continue
        if max_value is not None and v > max_value:
            print(f"Must be <= {max_value}.")
            continue
        return float(v)


def _prompt_choice(label: str, options: list[str], default: str, help_text: str) -> str:
    opt_str = "/".join(options)
    while True:
        raw = input(f"{label} [{default}] ({opt_str}) - {help_text}: ").strip()
        if raw == "":
            return default
        if raw in options:
            return raw
        print(f"Choose one of: {opt_str}")


def _prompt_bool(label: str, default: bool, help_text: str) -> bool:
    d = "y" if default else "n"
    while True:
        raw = input(f"{label} [{d}] (y/n) - {help_text}: ").strip().lower()
        if raw == "":
            return bool(default)
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Enter y or n.")


def _interactive_collect(args: argparse.Namespace) -> argparse.Namespace:
    print("Interactive mode: configure probe_scoring_dimensions options.")
    default_root = _resolve_default_doe_root()
    default_root_str = str(default_root) if default_root is not None else ""
    while True:
        root_raw = _prompt_text(
            "DOE root",
            default_root_str,
            "Path to DOE folder with doe_run_features.csv",
        )
        root = Path(root_raw)
        if (root / "doe_run_features.csv").exists():
            args.doe_root = root
            break
        print("Invalid DOE root (missing doe_run_features.csv).")

    args.objective_config = Path(
        _prompt_text(
            "Objective config",
            str(args.objective_config),
            "JSON with scoring thresholds/weights",
        )
    )
    dim_mode = _prompt_choice(
        "Dimensions mode",
        ["all", "custom"],
        "all" if args.dimensions is None else "custom",
        "Probe all dimensions or a custom comma-separated subset",
    )
    if dim_mode == "all":
        args.dimensions = None
    else:
        args.dimensions = _prompt_text(
            "Dimensions list",
            ",".join(PRIMARY_QUALITY_COMPONENT_KEYS),
            "Comma-separated quality keys",
        )

    args.dominant_weight = _prompt_float(
        "Dominant weight",
        float(args.dominant_weight),
        "Weight for focused dimension in each probe",
        min_value=0.0,
        max_value=1.0,
    )
    args.top_count = _prompt_int("Top count", int(args.top_count), "Top designs per dimension", min_value=1)
    args.mid_count = _prompt_int("Middle count", int(args.mid_count), "Middle designs per dimension", min_value=0)
    args.bottom_count = _prompt_int("Bottom count", int(args.bottom_count), "Bottom designs per dimension", min_value=0)
    args.top_pool = _prompt_int("Top pool", int(args.top_pool), "Candidate pool size used for diverse top picks", min_value=1)

    inferred_primary, inferred_robust = _infer_rules_from_doe_spec(Path(args.doe_root))
    args.primary_rule = _prompt_text(
        "Primary rule",
        str(args.primary_rule or inferred_primary),
        "Rule name used for scoring primary channel",
    )
    args.robust_rule = _prompt_text(
        "Robust rule",
        str(args.robust_rule or inferred_robust),
        "Rule name used for robustness channel",
    )

    render = _prompt_bool("Render summaries", not bool(args.no_render), "Create summary PDFs for selected runs")
    args.no_render = not render
    if render:
        args.summary_mode = _prompt_choice(
            "Summary mode",
            ["fast", "full"],
            str(args.summary_mode),
            "Rendering speed/detail",
        )
        args.summary_profile = _prompt_choice(
            "Summary profile",
            ["full", "debug_doe_compact", "thesis_core"],
            str(args.summary_profile),
            "Summary plot profile",
        )

    default_out = (
        str(args.out_dir)
        if args.out_dir is not None
        else str(Path(args.doe_root) / "analysis_post" / "scoring_dimension_probe")
    )
    args.out_dir = Path(_prompt_text("Output dir", default_out, "Folder for probe outputs"))
    return args


def _infer_rules_from_doe_spec(root: Path) -> tuple[str, str]:
    spec_path = root / "doe_spec.json"
    default_primary = "approval"
    default_robust = "utilitarian"
    if not spec_path.exists():
        return default_primary, default_robust
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception:
        return default_primary, default_robust
    if not isinstance(spec, dict):
        return default_primary, default_robust
    primary = str(spec.get("primary_rule_name", default_primary) or default_primary)
    robust = str(spec.get("robust_rule_name", default_robust) or default_robust)
    return primary, robust


def _required_seed_coverage(doe_root: Path, strict: bool) -> tuple[int | None, int | None]:
    if not strict:
        return None, None
    spec_path = doe_root / "doe_spec.json"
    if not spec_path.exists():
        return None, None
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    if not isinstance(spec, dict):
        return None, None
    seeds = spec.get("seeds", [])
    if not isinstance(seeds, list) or len(seeds) == 0:
        return None, None
    required_primary_runs = int(len(seeds))
    include_robustness = bool(spec.get("include_robustness", False))
    robust_every = int(spec.get("robust_every", 1))
    if include_robustness and robust_every == 1:
        return required_primary_runs, required_primary_runs
    return required_primary_runs, None


def _compute_qcw(target: str, dominant_weight: float) -> dict[str, float]:
    keys = list(PRIMARY_QUALITY_COMPONENT_KEYS)
    n = len(keys)
    dominant = float(np.clip(dominant_weight, 0.0, 1.0))
    rest = max(0.0, 1.0 - dominant)
    if n <= 1:
        return {target: 1.0}
    rest_each = rest / float(n - 1)
    out: dict[str, float] = {}
    for k in keys:
        out[k] = rest_each
    out[target] = dominant
    s = float(sum(out.values()))
    if s <= 0:
        return {k: 1.0 / float(n) for k in keys}
    return {k: float(v / s) for k, v in out.items()}


def _safe_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")


def _select_diverse_top(
    ranked: pd.DataFrame,
    *,
    top_pool: int,
    top_count: int,
) -> list[int]:
    if len(ranked) == 0 or top_count <= 0:
        return []
    pool = ranked.head(int(max(1, top_pool))).copy().reset_index(drop=True)
    if len(pool) <= top_count:
        return [int(x) for x in pool["design_id"].tolist()]

    feat_cols = [
        c
        for c in [
            "pass_rate",
            "quality_mean",
            "seed_robustness",
            "moderate_recovery_selector",
            "score_total",
        ]
        if c in pool.columns
    ]
    if not feat_cols:
        return [int(x) for x in pool.head(top_count)["design_id"].tolist()]

    X = pool[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (X - mu[None, :]) / sd[None, :]
    score = pd.to_numeric(pool["score_total"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    selected = [0]  # always include best rank.
    while len(selected) < int(top_count):
        best_j = None
        best_val = -1e18
        for j in range(len(pool)):
            if j in selected:
                continue
            dmin = min(float(np.linalg.norm(Z[j] - Z[k])) for k in selected)
            # slight high-score preference among equally diverse candidates
            val = dmin + 0.05 * float(score[j])
            if val > best_val:
                best_val = val
                best_j = j
        if best_j is None:
            break
        selected.append(int(best_j))
    return [int(pool.iloc[j]["design_id"]) for j in selected]


def _select_middle(ranked: pd.DataFrame, count: int) -> list[int]:
    if len(ranked) == 0 or count <= 0:
        return []
    n = len(ranked)
    if n <= count:
        return [int(x) for x in ranked["design_id"].tolist()]
    center = n // 2
    offs = np.linspace(-(count // 2), count // 2, num=count).round().astype(int)
    idxs = sorted({int(np.clip(center + int(o), 0, n - 1)) for o in offs})
    while len(idxs) < count:
        idxs.append(min(n - 1, idxs[-1] + 1))
        idxs = sorted(set(idxs))
    return [int(ranked.iloc[i]["design_id"]) for i in idxs[:count]]


def _select_bottom(ranked: pd.DataFrame, count: int) -> list[int]:
    if len(ranked) == 0 or count <= 0:
        return []
    return [int(x) for x in ranked.tail(count)["design_id"].tolist()]


def _pick_representative_run(
    gated: pd.DataFrame,
    *,
    design_id: int,
    primary_rule_name: str,
) -> pd.Series | None:
    sub = gated.loc[
        (pd.to_numeric(gated["design_id"], errors="coerce") == int(design_id))
        & (gated["rule_name"].astype(str) == str(primary_rule_name))
    ].copy()
    if len(sub) == 0:
        return None
    key_cols = [
        c
        for c in [
            "mean_turnout",
            "winner_change_rate_post_burnin",
            "group_turnout_range_mean",
            "puzzle_dominance_share_conflict",
            "moderate_recovery_strength_run",
        ]
        if c in sub.columns
    ]
    if key_cols:
        M = sub[key_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        med = np.nanmedian(M, axis=0)
        sd = np.nanstd(M, axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Z = (M - med[None, :]) / sd[None, :]
        sub["_typical_dist"] = np.linalg.norm(Z, axis=1)
    else:
        sub["_typical_dist"] = 0.0
    sub["_passes"] = pd.to_numeric(sub["passes_hard_gates"], errors="coerce").fillna(0).astype(int)
    sub = sub.sort_values(
        ["_passes", "_typical_dist", "seed"],
        ascending=[False, True, True],
    )
    return sub.iloc[0]


def _render_summary(run_dir: Path, out_dir: Path, *, mode: str, profile: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "scripts.generate_summary",
        "--run-dir",
        str(run_dir),
        "--out-dir",
        str(out_dir),
        "--mode",
        str(mode),
        "--closed",
        "--profile",
        str(profile),
    ]
    env = dict(**dict())
    env.update({"MPLCONFIGDIR": "/tmp/mpl"})
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    pdfs = sorted(out_dir.glob("global_summary*.pdf"))
    if p.returncode != 0 or not pdfs:
        return ""
    return str(pdfs[-1])


def _iter_dims(raw: str | None) -> Iterable[str]:
    if raw is None or str(raw).strip() == "":
        return list(PRIMARY_QUALITY_COMPONENT_KEYS)
    return [s.strip() for s in str(raw).split(",") if s.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Probe DOE scoring by forcing each quality component to dominate quality weights, "
            "then build top/mid/bottom HIL packets with optional summary PDFs."
        )
    )
    ap.add_argument("--doe-root", type=Path, default=None, help="DOE root (e.g. data/simulation_output/doe_XXXX).")
    ap.add_argument(
        "--objective-config",
        type=Path,
        default=Path("configs") / "doe_selection_objective_thesis_recovery_v1.json",
        help="Base objective config used for thresholds/top-level weights.",
    )
    ap.add_argument("--dimensions", type=str, default=None, help="Comma-separated quality keys to probe. Default: all.")
    ap.add_argument("--dominant-weight", type=float, default=0.90, help="Weight assigned to the focus dimension.")
    ap.add_argument("--top-count", type=int, default=10)
    ap.add_argument("--mid-count", type=int, default=3)
    ap.add_argument("--bottom-count", type=int, default=3)
    ap.add_argument("--top-pool", type=int, default=40, help="Top-rank pool used for diverse top selection.")
    ap.add_argument("--primary-rule", type=str, default=None, help="Default inferred from doe_spec.json.")
    ap.add_argument("--robust-rule", type=str, default=None, help="Default inferred from doe_spec.json.")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output root. Default: <DOE>/analysis_post/scoring_dimension_probe.")
    ap.add_argument("--no-render", action="store_true", help="Skip summary PDF rendering.")
    ap.add_argument("--summary-mode", choices=("fast", "full"), default="fast")
    ap.add_argument(
        "--summary-profile",
        choices=("full", "debug_doe_compact", "thesis_core"),
        default="debug_doe_compact",
    )
    ap.add_argument("--interactive", action="store_true", help="Force interactive prompts.")
    args = ap.parse_args()
    auto_interactive = bool(args.interactive) or len(sys.argv) == 1
    if auto_interactive:
        args = _interactive_collect(args)
    if args.doe_root is None:
        raise ValueError("Missing --doe-root. Run without flags to use interactive mode.")

    doe_root = Path(args.doe_root)
    rf_path = doe_root / "doe_run_features.csv"
    if not rf_path.exists():
        raise FileNotFoundError(f"Missing {rf_path}. Run scripts.score_doe first.")
    run_features = pd.read_csv(rf_path)

    objective = load_selection_objective(args.objective_config)
    thr = dict(objective["thresholds"])
    w = dict(objective["weights"])
    sw = dict(objective["stage_weights"])
    qcw_base = dict(objective.get("quality_component_weights") or {})
    strict = bool(objective.get("strict_completeness", True))

    inferred_primary, inferred_robust = _infer_rules_from_doe_spec(doe_root)
    primary_rule = str(args.primary_rule or inferred_primary)
    robust_rule = str(args.robust_rule or inferred_robust)
    req_primary, req_pairs = _required_seed_coverage(doe_root, strict=strict)

    gated = apply_hard_gates(
        run_features,
        max_all_abstain_stretch=float(thr["max_all_abstain_stretch"]),
        min_winner_changes_post_burnin=float(thr["min_winner_changes_post_burnin"]),
        min_winner_change_rate_post_burnin=float(thr["min_winner_change_rate_post_burnin"]),
        max_winner_change_rate_post_burnin=float(thr["max_winner_change_rate_post_burnin"]),
        min_group_turnout_range_mean=float(thr["min_group_turnout_range_mean"]),
        min_roll3_group_turnout_range_max=float(thr["min_roll3_group_turnout_range_max"]),
        min_roll20_group_turnout_range_max=float(thr["min_roll20_group_turnout_range_max"]),
        min_turnout_std=float(thr["min_turnout_std"]),
        min_gini_std=float(thr["min_gini_std"]),
        min_dist_std=float(thr["min_dist_std"]),
        min_winner_entropy_norm=float(thr["min_winner_entropy_norm"]),
        min_dist_nonzero_share=float(thr["min_dist_nonzero_share"]),
        min_competitive_step_share=float(thr["min_competitive_step_share"]),
        min_mean_turnout=float(thr["min_mean_turnout"]),
        max_mean_turnout=float(thr["max_mean_turnout"]),
        min_puzzle_conflict_step_share_for_gate=float(thr["min_puzzle_conflict_step_share_for_gate"]),
        max_puzzle_dominance_share_conflict=float(thr["max_puzzle_dominance_share_conflict"]),
        min_power_recovery_share_conflict=float(thr["min_power_recovery_share_conflict"]),
    )

    out_root = (
        Path(args.out_dir)
        if args.out_dir is not None
        else (doe_root / "analysis_post" / "scoring_dimension_probe")
    )
    out_root.mkdir(parents=True, exist_ok=True)

    master_rows: list[dict] = []
    dims = list(_iter_dims(args.dimensions))
    for dim in dims:
        if dim not in PRIMARY_QUALITY_COMPONENT_KEYS:
            raise ValueError(f"Unknown dimension: {dim}")
        dim_slug = _safe_slug(dim)
        dim_dir = out_root / dim_slug
        dim_dir.mkdir(parents=True, exist_ok=True)

        qcw = _compute_qcw(dim, dominant_weight=float(args.dominant_weight))
        if qcw_base:
            # Keep same key-space and only adjust distribution among known keys.
            qcw = {k: qcw[k] for k in qcw_base.keys() if k in qcw}
            s = float(sum(qcw.values()))
            if s > 0.0:
                qcw = {k: float(v / s) for k, v in qcw.items()}

        scores = score_designs(
            gated,
            primary_rule_name=primary_rule,
            robust_rule_name=robust_rule,
            weights=w,
            stage_weights=sw,
            required_primary_runs=req_primary,
            required_matched_seed_pairs=req_pairs,
            min_winner_entropy_norm=float(thr["min_winner_entropy_norm"]),
            min_competitive_step_share=float(thr["min_competitive_step_share"]),
            puzzle_dominance_share_score_low=float(thr["puzzle_dominance_share_score_low"]),
            puzzle_dominance_share_score_high=float(thr["puzzle_dominance_share_score_high"]),
            moderate_selector_quality_gate_zero_at=float(thr["moderate_selector_quality_gate_zero_at"]),
            moderate_selector_quality_gate_good_min=float(thr["moderate_selector_quality_gate_good_min"]),
            turnout_start_score_low=float(thr["turnout_start_score_low"]),
            turnout_start_score_high=float(thr["turnout_start_score_high"]),
            turnout_end_score_low=float(thr["turnout_end_score_low"]),
            turnout_end_score_high=float(thr["turnout_end_score_high"]),
            turnout_drop_score_good_max=float(thr["turnout_drop_score_good_max"]),
            turnout_drop_score_zero_at=float(thr["turnout_drop_score_zero_at"]),
            turnout_decline_score_good_max=float(thr["turnout_decline_score_good_max"]),
            turnout_decline_score_zero_at=float(thr["turnout_decline_score_zero_at"]),
            turnout_outside_band_share_good_max=float(thr["turnout_outside_band_share_good_max"]),
            turnout_outside_band_share_zero_at=float(thr["turnout_outside_band_share_zero_at"]),
            participation_q_delta_mean_abs_good_max=float(thr["participation_q_delta_mean_abs_good_max"]),
            participation_q_delta_mean_abs_zero_at=float(thr["participation_q_delta_mean_abs_zero_at"]),
            participation_q_delta_late_mean_abs_good_max=float(thr["participation_q_delta_late_mean_abs_good_max"]),
            participation_q_delta_late_mean_abs_zero_at=float(thr["participation_q_delta_late_mean_abs_zero_at"]),
            quality_component_weights=qcw,
        )
        scores = scores.reset_index(drop=True)
        scores.insert(0, "rank", np.arange(1, len(scores) + 1, dtype=int))
        scores.to_csv(dim_dir / "design_scores.csv", index=False)

        top_ids = _select_diverse_top(
            scores,
            top_pool=int(args.top_pool),
            top_count=int(args.top_count),
        )
        mid_ids = _select_middle(scores, int(args.mid_count))
        bottom_ids = _select_bottom(scores, int(args.bottom_count))
        buckets = [("top", top_ids), ("middle", mid_ids), ("bottom", bottom_ids)]

        dim_rows: list[dict] = []
        for bucket, ids in buckets:
            bucket_dir = dim_dir / bucket
            bucket_dir.mkdir(parents=True, exist_ok=True)
            for pos, design_id in enumerate(ids, start=1):
                ds = scores.loc[pd.to_numeric(scores["design_id"], errors="coerce") == int(design_id)]
                if len(ds) == 0:
                    continue
                drow = ds.iloc[0]
                rrow = _pick_representative_run(gated, design_id=int(design_id), primary_rule_name=primary_rule)
                if rrow is None:
                    continue
                run_dir = Path(str(rrow["run_dir"]))
                if not run_dir.is_absolute():
                    run_dir = Path.cwd() / run_dir
                seed = int(pd.to_numeric(rrow["seed"], errors="coerce"))
                run_out = bucket_dir / f"{bucket}_p{pos:02d}_design_{int(design_id):04d}_seed_{seed:05d}"
                pdf_path = ""
                if not bool(args.no_render):
                    pdf_path = _render_summary(
                        run_dir,
                        run_out,
                        mode=str(args.summary_mode),
                        profile=str(args.summary_profile),
                    )
                row = {
                    "dimension": dim,
                    "bucket": bucket,
                    "bucket_pos": int(pos),
                    "design_id": int(design_id),
                    "seed": int(seed),
                    "score_total": float(drow["score_total"]),
                    "pass_rate": float(drow["pass_rate"]),
                    "quality_mean": float(drow["quality_mean"]),
                    "seed_robustness": float(drow["seed_robustness"]),
                    "moderate_recovery_selector": float(drow.get("moderate_recovery_selector", 0.0)),
                    "run_dir": str(run_dir),
                    "summary_pdf_path": str(pdf_path),
                    "human_verdict": "",
                    "human_feedback": "",
                    "human_confidence": "",
                    "adjustment_hint": "",
                }
                dim_rows.append(row)
                master_rows.append(row)

        pd.DataFrame(dim_rows).to_csv(dim_dir / "hil_table.csv", index=False)

    pd.DataFrame(master_rows).to_csv(out_root / "hil_table_all_dimensions.csv", index=False)
    (out_root / "probe_meta.json").write_text(
        json.dumps(
            {
                "doe_root": str(doe_root),
                "objective_config": str(args.objective_config),
                "dimensions": dims,
                "dominant_weight": float(args.dominant_weight),
                "top_count": int(args.top_count),
                "mid_count": int(args.mid_count),
                "bottom_count": int(args.bottom_count),
                "top_pool": int(args.top_pool),
                "rendered_summaries": not bool(args.no_render),
                "summary_mode": str(args.summary_mode),
                "summary_profile": str(args.summary_profile),
                "primary_rule": str(primary_rule),
                "robust_rule": str(robust_rule),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote: {out_root}")
    print(f"Wrote: {out_root / 'hil_table_all_dimensions.csv'}")


if __name__ == "__main__":
    main()
