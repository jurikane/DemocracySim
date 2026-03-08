from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RecoveryThresholds:
    min_group_size: int
    low_turnout_max_rel: float
    low_min_steps: int
    dominant_turnout_min_rel: float
    altruism_min_share: float
    altruism_min_steps: int
    recovery_window_steps: int
    selflead_min_steps: int
    rebound_min_turnout_rel: float
    rebound_min_steps: int
    dominant_drop_min_frac: float
    # Moderate recovery tier (pairwise turnout-share cannibalization + reversal).
    moderate_takeover_loss_frac_min: float
    moderate_takeover_capture_frac_min: float
    moderate_rebound_loss_frac_min: float
    moderate_rebound_share_vs_a_peak_min: float
    moderate_min_steps: int
    moderate_min_peak_share: float


def _resolve_default_roots() -> list[Path]:
    base = Path("data") / "simulation_output"
    roots = sorted(p for p in base.glob("doe_*") if p.is_dir())
    if not roots:
        raise FileNotFoundError("No DOE roots found under data/simulation_output (expected doe_*).")
    return [roots[-1]]


def _longest_true_streak(mask: np.ndarray) -> int:
    best = 0
    cur = 0
    for v in mask:
        if bool(v):
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def _first_true_streak(mask: np.ndarray, min_len: int) -> tuple[int, int] | None:
    cur = 0
    start = 0
    for i, v in enumerate(mask):
        if bool(v):
            if cur == 0:
                start = i
            cur += 1
            if cur >= int(min_len):
                return start, i
        else:
            cur = 0
    return None


def _iter_run_dirs(doe_root: Path, rule_name: str) -> Iterable[Path]:
    pattern = f"design_*/rule_{rule_name}/seed_*/run_0"
    yield from sorted(doe_root.glob(pattern))


def _build_turnout_and_vote_matrices(
    run_dir: Path,
    *,
    min_group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], np.ndarray, np.ndarray] | None:
    agents_path = run_dir / "agents.parquet"
    votes_path = run_dir / "votes.parquet"
    if not agents_path.exists() or not votes_path.exists():
        return None

    agents = pd.read_parquet(
        agents_path,
        columns=["step", "agent_id", "personality_group_idx", "participating"],
    )
    votes = pd.read_parquet(
        votes_path,
        columns=["step", "agent_id", "voted_altruistically"],
    )

    if agents.empty:
        return None

    agent_group = agents[["agent_id", "personality_group_idx"]].drop_duplicates("agent_id")
    group_sizes = agent_group.groupby("personality_group_idx").size().sort_index()
    groups = group_sizes[group_sizes >= int(min_group_size)].index.tolist()
    if len(groups) < 2:
        return None

    all_steps = np.arange(int(agents["step"].min()), int(agents["step"].max()) + 1, dtype=int)
    step0 = int(all_steps[0])
    n_steps = int(all_steps.size)
    n_groups = int(len(groups))
    gidx = {int(g): i for i, g in enumerate(groups)}

    turnout_rel = np.zeros((n_steps, n_groups), dtype=np.float64)
    participants = agents[agents["participating"] == True].groupby(["step", "personality_group_idx"]).size()
    for (step, group), n_part in participants.items():
        group = int(group)
        if group not in gidx:
            continue
        i = int(step) - step0
        j = gidx[group]
        turnout_rel[i, j] = float(n_part) / float(group_sizes[group])

    merged_votes = votes.merge(agent_group, on="agent_id", how="left")
    merged_votes = merged_votes[merged_votes["personality_group_idx"].isin(groups)]
    g = merged_votes.groupby(["step", "personality_group_idx"])
    total_votes = g.size()
    altru_votes = g["voted_altruistically"].sum()
    self_votes = total_votes - altru_votes

    altru_share = np.zeros((n_steps, n_groups), dtype=np.float64)
    self_vote_counts = np.zeros((n_steps, n_groups), dtype=np.float64)
    for (step, group), n_total in total_votes.items():
        group = int(group)
        i = int(step) - step0
        j = gidx[group]
        n_total_f = float(n_total)
        n_self = float(self_votes.loc[(step, group)])
        n_altru = float(altru_votes.loc[(step, group)])
        self_vote_counts[i, j] = n_self
        altru_share[i, j] = n_altru / n_total_f if n_total_f > 0.0 else 0.0

    group_sizes_vec = np.asarray([float(group_sizes[int(g)]) for g in groups], dtype=np.float64)
    return turnout_rel, altru_share, self_vote_counts, [int(g) for g in groups], all_steps, group_sizes_vec


def _scan_run_moderate(
    *,
    run_dir: Path,
    thresholds: RecoveryThresholds,
) -> list[dict]:
    payload = _build_turnout_and_vote_matrices(run_dir, min_group_size=thresholds.min_group_size)
    if payload is None:
        return []
    turnout_rel, _, _, groups, all_steps, group_sizes = payload

    # Convert within-group turnout to total-turnout share per step.
    turnout_counts = turnout_rel * group_sizes.reshape(1, -1)
    total_counts = turnout_counts.sum(axis=1, keepdims=True)
    safe_total = np.where(total_counts > 1e-12, total_counts, 1.0)
    turnout_share = turnout_counts / safe_total

    n_steps, n_groups = turnout_share.shape
    if n_groups < 2 or n_steps < 4:
        return []

    events: list[dict] = []
    for a_j, a_group in enumerate(groups):
        for b_j, b_group in enumerate(groups):
            if a_j == b_j:
                continue

            # Phase 1: A eats B share (B loses at least X% from B peak; A captures enough of that loss).
            b_peak_idx = int(np.argmax(turnout_share[:, b_j]))
            b_peak = float(turnout_share[b_peak_idx, b_j])
            if b_peak < float(thresholds.moderate_min_peak_share):
                continue
            a_at_b_peak = float(turnout_share[b_peak_idx, a_j])

            after_b_peak = np.arange(b_peak_idx + 1, n_steps, dtype=int)
            if after_b_peak.size == 0:
                continue
            b_vals = turnout_share[after_b_peak, b_j]
            a_vals = turnout_share[after_b_peak, a_j]
            b_loss = b_peak - b_vals
            a_gain = a_vals - a_at_b_peak
            phase1_mask = (
                (b_loss >= float(thresholds.moderate_takeover_loss_frac_min) * b_peak)
                & (a_gain >= float(thresholds.moderate_takeover_capture_frac_min) * b_loss)
            )
            phase1_streak = _first_true_streak(phase1_mask, int(thresholds.moderate_min_steps))
            if phase1_streak is None:
                continue
            s1, e1 = phase1_streak
            t1_start = int(after_b_peak[s1])
            t1_end = int(after_b_peak[e1])

            # Phase 2: B eats A back (A loses >= Y% from post-phase1 A peak; B reaches >= Z * A_peak).
            post1 = np.arange(t1_end + 1, n_steps, dtype=int)
            if post1.size == 0:
                continue
            a_post = turnout_share[post1, a_j]
            a_peak_rel_idx = int(np.argmax(a_post))
            t_a_peak = int(post1[a_peak_rel_idx])
            a_peak = float(turnout_share[t_a_peak, a_j])
            if a_peak < float(thresholds.moderate_min_peak_share):
                continue

            after_a_peak = np.arange(t_a_peak + 1, n_steps, dtype=int)
            if after_a_peak.size == 0:
                continue
            a_after = turnout_share[after_a_peak, a_j]
            b_after = turnout_share[after_a_peak, b_j]
            phase2_mask = (
                ((a_peak - a_after) >= float(thresholds.moderate_rebound_loss_frac_min) * a_peak)
                & (b_after >= float(thresholds.moderate_rebound_share_vs_a_peak_min) * a_peak)
            )
            phase2_streak = _first_true_streak(phase2_mask, int(thresholds.moderate_min_steps))
            if phase2_streak is None:
                continue
            s2, e2 = phase2_streak
            t2_start = int(after_a_peak[s2])
            t2_end = int(after_a_peak[e2])

            event = {
                "run_path": str(run_dir),
                "a_group": int(a_group),
                "b_group": int(b_group),
                "b_peak_step": int(all_steps[b_peak_idx]),
                "b_peak_share": float(b_peak),
                "phase1_start_step": int(all_steps[t1_start]),
                "phase1_end_step": int(all_steps[t1_end]),
                "b_share_at_phase1_end": float(turnout_share[t1_end, b_j]),
                "a_share_at_b_peak": float(a_at_b_peak),
                "a_share_at_phase1_end": float(turnout_share[t1_end, a_j]),
                "a_peak_step": int(all_steps[t_a_peak]),
                "a_peak_share": float(a_peak),
                "phase2_start_step": int(all_steps[t2_start]),
                "phase2_end_step": int(all_steps[t2_end]),
                "a_share_at_phase2_end": float(turnout_share[t2_end, a_j]),
                "b_share_at_phase2_end": float(turnout_share[t2_end, b_j]),
                "b_loss_from_peak_frac": float((b_peak - turnout_share[t1_end, b_j]) / max(b_peak, 1e-12)),
                "a_loss_from_peak_frac": float((a_peak - turnout_share[t2_end, a_j]) / max(a_peak, 1e-12)),
                "moderate_event": True,
            }
            events.append(event)

    return events


def _scan_run(
    *,
    run_dir: Path,
    thresholds: RecoveryThresholds,
) -> tuple[dict | None, list[dict]]:
    payload = _build_turnout_and_vote_matrices(run_dir, min_group_size=thresholds.min_group_size)
    if payload is None:
        return None, []
    turnout_rel, altru_share, self_vote_counts, groups, all_steps, _ = payload

    n_steps, _ = turnout_rel.shape
    best: dict | None = None
    best_score = -1e18
    strict_events: list[dict] = []

    for weak_j, weak_group in enumerate(groups):
        low_mask = turnout_rel[:, weak_j] <= float(thresholds.low_turnout_max_rel)
        s = 0
        while s < n_steps:
            if not low_mask[s]:
                s += 1
                continue
            e = s
            while e < n_steps and low_mask[e]:
                e += 1
            low_len = e - s
            if low_len >= int(thresholds.low_min_steps):
                dom_mean_by_group = turnout_rel[s:e, :].mean(axis=0)
                dom_j = int(np.argmax(dom_mean_by_group))
                dom_group = int(groups[dom_j])
                dom_base = float(dom_mean_by_group[dom_j])
                if dom_j != weak_j and dom_base >= float(thresholds.dominant_turnout_min_rel):
                    fut_lo = int(e)
                    fut_hi = min(int(n_steps), int(e + thresholds.recovery_window_steps))
                    if fut_hi > fut_lo:
                        d_altru_streak = _longest_true_streak(
                            altru_share[fut_lo:fut_hi, dom_j] >= float(thresholds.altruism_min_share)
                        )
                        w_selflead_streak = _longest_true_streak(
                            self_vote_counts[fut_lo:fut_hi, weak_j] > self_vote_counts[fut_lo:fut_hi, dom_j]
                        )
                        w_rebound_streak = _longest_true_streak(
                            turnout_rel[fut_lo:fut_hi, weak_j] >= float(thresholds.rebound_min_turnout_rel)
                        )
                        w_rebound_max = float(np.max(turnout_rel[fut_lo:fut_hi, weak_j]))
                        d_min_after = float(np.min(turnout_rel[fut_lo:fut_hi, dom_j]))
                        d_drop_frac = (dom_base - d_min_after) / max(dom_base, 1e-12)

                        has_alt = d_altru_streak >= int(thresholds.altruism_min_steps)
                        has_lead = w_selflead_streak >= int(thresholds.selflead_min_steps)
                        has_rebound = w_rebound_streak >= int(thresholds.rebound_min_steps)
                        has_drop = d_drop_frac >= float(thresholds.dominant_drop_min_frac)
                        strict = bool(has_alt and has_lead and has_rebound and has_drop)

                        score = (
                            2.0 * float(has_alt)
                            + 2.0 * float(has_lead)
                            + 2.0 * float(has_rebound)
                            + 2.0 * float(has_drop)
                            + 4.0 * max(0.0, w_rebound_max - float(thresholds.rebound_min_turnout_rel))
                            + 3.0 * max(0.0, d_drop_frac - float(thresholds.dominant_drop_min_frac))
                            + 2.0 * max(0.0, dom_base - float(thresholds.dominant_turnout_min_rel))
                            + 0.2 * max(0.0, float(d_altru_streak - int(thresholds.altruism_min_steps)))
                            + 0.2 * max(0.0, float(w_rebound_streak - int(thresholds.rebound_min_steps)))
                        )

                        rec = {
                            "run_path": str(run_dir),
                            "w_group": int(weak_group),
                            "d_group": int(dom_group),
                            "low_start": int(all_steps[s]),
                            "low_end": int(all_steps[e - 1]),
                            "low_len": int(low_len),
                            "d_base_turnout_rel": float(dom_base),
                            "d_altru_streak": int(d_altru_streak),
                            "w_selflead_streak": int(w_selflead_streak),
                            "w_rebound_streak": int(w_rebound_streak),
                            "w_rebound_max_rel": float(w_rebound_max),
                            "d_drop_frac": float(d_drop_frac),
                            "has_alt": bool(has_alt),
                            "has_lead": bool(has_lead),
                            "has_rebound": bool(has_rebound),
                            "has_drop": bool(has_drop),
                            "strict_event": bool(strict),
                            "score": float(score),
                        }
                        if score > best_score:
                            best_score = score
                            best = rec
                        if strict:
                            strict_events.append(rec)
            s = e
    return best, strict_events


def _extract_design_seed(run_path: str) -> tuple[int | None, int | None]:
    p = Path(run_path)
    try:
        design_id = int(p.parts[-4].split("_")[1])
        seed = int(p.parts[-2].split("_")[1])
        return design_id, seed
    except (IndexError, ValueError):
        return None, None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Scan DOE run parquet files for strict lock-in recovery events and near-miss runs "
            "(approval by default)."
        )
    )
    ap.add_argument(
        "--doe-root",
        type=Path,
        action="append",
        default=None,
        help="DOE root directory. Repeat to scan multiple DOE roots. Default: latest doe_* under data/simulation_output.",
    )
    ap.add_argument("--rule-name", type=str, default="approval", help="Rule folder suffix to scan (default: approval).")
    ap.add_argument("--max-runs", type=int, default=0, help="Optional cap for quick smoke tests (0 = all runs).")
    ap.add_argument("--progress-every", type=int, default=400, help="Print progress every N scanned runs.")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data") / "simulation_output" / "analysis_post",
        help="Output directory for CSV/JSON artifacts.",
    )
    ap.add_argument("--out-prefix", type=str, default="strict_recovery_scan", help="Output filename prefix.")

    # strict thresholds
    ap.add_argument("--min-group-size", type=int, default=6)
    ap.add_argument("--low-turnout-max-rel", type=float, default=0.05)
    ap.add_argument("--low-min-steps", type=int, default=4)
    ap.add_argument("--dominant-turnout-min-rel", type=float, default=0.45)
    ap.add_argument("--altruism-min-share", type=float, default=0.90)
    ap.add_argument("--altruism-min-steps", type=int, default=4)
    ap.add_argument("--recovery-window-steps", type=int, default=70)
    ap.add_argument("--selflead-min-steps", type=int, default=3)
    ap.add_argument("--rebound-min-turnout-rel", type=float, default=0.22)
    ap.add_argument("--rebound-min-steps", type=int, default=6)
    ap.add_argument("--dominant-drop-min-frac", type=float, default=0.18)
    # moderate tier thresholds
    ap.add_argument("--moderate-takeover-loss-frac-min", type=float, default=0.50)
    ap.add_argument("--moderate-takeover-capture-frac-min", type=float, default=0.50)
    ap.add_argument("--moderate-rebound-loss-frac-min", type=float, default=0.50)
    ap.add_argument("--moderate-rebound-share-vs-a-peak-min", type=float, default=0.50)
    ap.add_argument("--moderate-min-steps", type=int, default=3)
    ap.add_argument("--moderate-min-peak-share", type=float, default=0.00)
    args = ap.parse_args()

    doe_roots = [Path(p) for p in (args.doe_root or _resolve_default_roots())]
    thresholds = RecoveryThresholds(
        min_group_size=int(args.min_group_size),
        low_turnout_max_rel=float(args.low_turnout_max_rel),
        low_min_steps=int(args.low_min_steps),
        dominant_turnout_min_rel=float(args.dominant_turnout_min_rel),
        altruism_min_share=float(args.altruism_min_share),
        altruism_min_steps=int(args.altruism_min_steps),
        recovery_window_steps=int(args.recovery_window_steps),
        selflead_min_steps=int(args.selflead_min_steps),
        rebound_min_turnout_rel=float(args.rebound_min_turnout_rel),
        rebound_min_steps=int(args.rebound_min_steps),
        dominant_drop_min_frac=float(args.dominant_drop_min_frac),
        moderate_takeover_loss_frac_min=float(args.moderate_takeover_loss_frac_min),
        moderate_takeover_capture_frac_min=float(args.moderate_takeover_capture_frac_min),
        moderate_rebound_loss_frac_min=float(args.moderate_rebound_loss_frac_min),
        moderate_rebound_share_vs_a_peak_min=float(args.moderate_rebound_share_vs_a_peak_min),
        moderate_min_steps=int(args.moderate_min_steps),
        moderate_min_peak_share=float(args.moderate_min_peak_share),
    )

    near_rows: list[dict] = []
    strict_rows: list[dict] = []
    moderate_rows: list[dict] = []
    summary_rows: list[dict] = []
    total_scanned = 0

    for root in doe_roots:
        run_dirs = list(_iter_run_dirs(root, rule_name=str(args.rule_name)))
        if int(args.max_runs) > 0:
            run_dirs = run_dirs[: int(args.max_runs)]
        strict_count_before = len(strict_rows)
        moderate_count_before = len(moderate_rows)
        near_count_before = len(near_rows)

        print(f"[scan] DOE root={root} runs={len(run_dirs)} rule={args.rule_name}")
        for i, run_dir in enumerate(run_dirs, start=1):
            total_scanned += 1
            best, strict_events = _scan_run(run_dir=run_dir, thresholds=thresholds)
            moderate_events = _scan_run_moderate(run_dir=run_dir, thresholds=thresholds)
            if best is not None:
                design_id, seed = _extract_design_seed(best["run_path"])
                best["doe_root"] = str(root)
                best["design_id"] = design_id
                best["seed"] = seed
                near_rows.append(best)
            for ev in strict_events:
                design_id, seed = _extract_design_seed(ev["run_path"])
                ev["doe_root"] = str(root)
                ev["design_id"] = design_id
                ev["seed"] = seed
                strict_rows.append(ev)
            for ev in moderate_events:
                design_id, seed = _extract_design_seed(ev["run_path"])
                ev["doe_root"] = str(root)
                ev["design_id"] = design_id
                ev["seed"] = seed
                moderate_rows.append(ev)

            if int(args.progress_every) > 0 and (i % int(args.progress_every) == 0):
                print(f"[scan] {root.name}: {i}/{len(run_dirs)} strict_events={len(strict_rows) - strict_count_before}")

        near_slice = near_rows[near_count_before:]
        strict_slice = strict_rows[strict_count_before:]
        moderate_slice = moderate_rows[moderate_count_before:]
        summary_rows.append(
            {
                "doe_root": str(root),
                "doe_name": root.name,
                "runs_scanned": int(len(run_dirs)),
                "near_rows": int(len(near_slice)),
                "strict_event_rows": int(len(strict_slice)),
                "strict_run_count": int(len({r['run_path'] for r in strict_slice})),
                "moderate_event_rows": int(len(moderate_slice)),
                "moderate_run_count": int(len({r['run_path'] for r in moderate_slice})),
                "near_score_mean": float(np.mean([float(r["score"]) for r in near_slice])) if near_slice else float("nan"),
                "near_score_p95": float(np.quantile([float(r["score"]) for r in near_slice], 0.95)) if near_slice else float("nan"),
                "near_rebound_max": float(np.max([float(r["w_rebound_max_rel"]) for r in near_slice])) if near_slice else float("nan"),
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.out_prefix).strip()
    near_csv = out_dir / f"{prefix}_near_misses.csv"
    strict_csv = out_dir / f"{prefix}_strict_events.csv"
    moderate_csv = out_dir / f"{prefix}_moderate_events.csv"
    summary_csv = out_dir / f"{prefix}_summary.csv"
    summary_json = out_dir / f"{prefix}_summary.json"

    near_df = pd.DataFrame(near_rows)
    if not near_df.empty:
        near_df = near_df.sort_values("score", ascending=False).reset_index(drop=True)
    strict_df = pd.DataFrame(strict_rows)
    if not strict_df.empty:
        strict_df = strict_df.sort_values("score", ascending=False).reset_index(drop=True)
    moderate_df = pd.DataFrame(moderate_rows)
    if not moderate_df.empty:
        moderate_df = moderate_df.sort_values(
            ["doe_root", "design_id", "seed", "phase2_end_step"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows)

    near_df.to_csv(near_csv, index=False)
    strict_df.to_csv(strict_csv, index=False)
    moderate_df.to_csv(moderate_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(
        json.dumps(
            {
                "rule_name": str(args.rule_name),
                "doe_roots": [str(r) for r in doe_roots],
                "runs_scanned_total": int(total_scanned),
                "thresholds": {
                    "min_group_size": thresholds.min_group_size,
                    "low_turnout_max_rel": thresholds.low_turnout_max_rel,
                    "low_min_steps": thresholds.low_min_steps,
                    "dominant_turnout_min_rel": thresholds.dominant_turnout_min_rel,
                    "altruism_min_share": thresholds.altruism_min_share,
                    "altruism_min_steps": thresholds.altruism_min_steps,
                    "recovery_window_steps": thresholds.recovery_window_steps,
                    "selflead_min_steps": thresholds.selflead_min_steps,
                    "rebound_min_turnout_rel": thresholds.rebound_min_turnout_rel,
                    "rebound_min_steps": thresholds.rebound_min_steps,
                    "dominant_drop_min_frac": thresholds.dominant_drop_min_frac,
                    "moderate_takeover_loss_frac_min": thresholds.moderate_takeover_loss_frac_min,
                    "moderate_takeover_capture_frac_min": thresholds.moderate_takeover_capture_frac_min,
                    "moderate_rebound_loss_frac_min": thresholds.moderate_rebound_loss_frac_min,
                    "moderate_rebound_share_vs_a_peak_min": thresholds.moderate_rebound_share_vs_a_peak_min,
                    "moderate_min_steps": thresholds.moderate_min_steps,
                    "moderate_min_peak_share": thresholds.moderate_min_peak_share,
                },
                "summary": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[done] scanned runs={total_scanned}")
    print(f"[done] near misses: {near_csv} rows={len(near_df)}")
    print(f"[done] strict events: {strict_csv} rows={len(strict_df)}")
    print(f"[done] moderate events: {moderate_csv} rows={len(moderate_df)}")
    print(f"[done] summary: {summary_csv}")
    print(f"[done] summary json: {summary_json}")
    if len(strict_df) == 0:
        print("[done] strict_event_rows=0 (no runs satisfied all strict recovery conditions).")


if __name__ == "__main__":
    main()
