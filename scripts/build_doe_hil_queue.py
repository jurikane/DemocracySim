from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


_SEED_RE = re.compile(r"seed_(\d+)")
_RULE_RE = re.compile(r"/rule_([^/]+)/")


def _resolve_doe_root(raw: str) -> Path:
    p = Path(str(raw))
    if p.exists():
        return p
    candidate = Path("data") / "simulation_output" / str(raw)
    if candidate.exists():
        return candidate
    return p


def _pick_representative_run(run_df: pd.DataFrame, *, bucket: str) -> pd.Series:
    # Deterministic selection of one run row for a design.
    df = run_df.copy()
    if bucket == "bottom":
        sort_cols = [
            "passes_hard_gates",
            "roll20_group_turnout_range_max",
            "competitive_step_share",
            "winner_entropy_norm",
        ]
        asc = [True, True, True, True]
    else:
        sort_cols = [
            "passes_hard_gates",
            "roll20_group_turnout_range_max",
            "competitive_step_share",
            "winner_entropy_norm",
        ]
        asc = [False, False, False, False]
    existing_cols = [c for c in sort_cols if c in df.columns]
    if "seed" in df.columns:
        existing_cols = existing_cols + ["seed"]
        existing_asc = [asc[sort_cols.index(c)] for c in existing_cols if c in sort_cols] + [True]
    else:
        existing_asc = [asc[sort_cols.index(c)] for c in existing_cols]
    if existing_cols:
        df = df.sort_values(existing_cols, ascending=existing_asc, kind="mergesort")
    return df.iloc[0]


def _seed_rule_from_run_dir(run_dir: str) -> tuple[int | None, str | None]:
    text = str(run_dir or "")
    if not text:
        return None, None
    seed_m = _SEED_RE.search(text)
    rule_m = _RULE_RE.search(text)
    seed = int(seed_m.group(1)) if seed_m else None
    rule_name = rule_m.group(1) if rule_m else None
    return seed, rule_name


def _sample_buckets(design_df: pd.DataFrame, *, n_per_bucket: int) -> dict[str, pd.DataFrame]:
    ranked = design_df.sort_values(["score_total", "pass_rate", "design_id"], ascending=[False, False, True]).reset_index(drop=True)
    n = len(ranked)
    if n == 0:
        return {"top": ranked, "mid": ranked, "bottom": ranked}

    k = min(int(n_per_bucket), n)
    top = ranked.head(k).copy()
    bottom = ranked.tail(k).sort_values(["score_total", "pass_rate", "design_id"], ascending=[True, True, True]).copy()

    # Prefer "mid" from viable/near-viable designs if available, otherwise fallback to full ranked list.
    viable = ranked[(pd.to_numeric(ranked.get("score_total"), errors="coerce").fillna(0.0) > 0.0) | (pd.to_numeric(ranked.get("pass_rate"), errors="coerce").fillna(0.0) > 0.0)].copy()
    mid_pool = viable.reset_index(drop=True) if len(viable) >= k else ranked
    m = len(mid_pool)
    mid_center = m // 2
    half = k // 2
    start = max(0, mid_center - half)
    end = min(m, start + k)
    start = max(0, end - k)
    mid = mid_pool.iloc[start:end].copy()

    return {"top": top, "mid": mid, "bottom": bottom}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build top/mid/bottom HIL queue CSV from a scored DOE root.")
    parser.add_argument("--doe-root", required=True, help="DOE root path or name under data/simulation_output.")
    parser.add_argument("--out-csv", type=Path, default=None, help="Output CSV path (default: <doe-root>/doe_hil_queue_v1.csv)")
    parser.add_argument("--per-bucket", type=int, default=10, help="Rows per bucket (top/mid/bottom).")
    parser.add_argument(
        "--rule-name",
        type=str,
        default="approval",
        help="Rule to use for representative run rows (default: approval).",
    )
    args = parser.parse_args()

    doe_root = _resolve_doe_root(args.doe_root)
    if not doe_root.exists():
        raise FileNotFoundError(f"DOE root not found: {doe_root}")

    design_scores_path = doe_root / "doe_design_scores.csv"
    run_features_path = doe_root / "doe_run_features.csv"
    if not design_scores_path.exists() or not run_features_path.exists():
        raise FileNotFoundError("DOE root must contain doe_design_scores.csv and doe_run_features.csv (score it first).")

    design_df = pd.read_csv(design_scores_path)
    run_df = pd.read_csv(run_features_path)
    if "rule_name" in run_df.columns:
        run_df = run_df[run_df["rule_name"].astype(str) == str(args.rule_name)].copy()
    if run_df.empty:
        raise RuntimeError(f"No run features found for rule_name={args.rule_name!r}")

    buckets = _sample_buckets(design_df, n_per_bucket=int(args.per_bucket))
    rows: list[dict[str, Any]] = []
    doe_root_value = str(doe_root)
    doe_root_name = doe_root.name

    for bucket_name in ("top", "mid", "bottom"):
        bucket_df = buckets[bucket_name]
        for _, drow in bucket_df.iterrows():
            did = int(drow["design_id"])
            runs = run_df[run_df["design_id"].astype(int) == did]
            if runs.empty:
                rep_run_dir = ""
                rep_seed: int | None = None
                rep_rule: str | None = None
            else:
                rep = _pick_representative_run(runs, bucket=bucket_name)
                rep_run_dir = str(rep.get("run_dir", ""))
                rep_seed = None
                rep_rule = None
                if "seed" in rep.index and pd.notna(rep.get("seed")):
                    try:
                        rep_seed = int(rep.get("seed"))  # type: ignore[arg-type]
                    except Exception:
                        rep_seed = None
                if "rule_name" in rep.index and pd.notna(rep.get("rule_name")):
                    rep_rule = str(rep.get("rule_name"))
                if rep_seed is None or rep_rule is None:
                    parsed_seed, parsed_rule = _seed_rule_from_run_dir(rep_run_dir)
                    rep_seed = rep_seed if rep_seed is not None else parsed_seed
                    rep_rule = rep_rule if rep_rule is not None else parsed_rule
            rows.append(
                {
                    "doe_root": doe_root_name,
                    "bucket": bucket_name,
                    "design_id": did,
                    "score_total": float(drow.get("score_total", float("nan"))),
                    "pass_rate": float(drow.get("pass_rate", float("nan"))),
                    "representative_run_dir": rep_run_dir,
                    "seed": rep_seed,
                    "rule_name": rep_rule,
                    "ai_interpretation": "",
                    "human_feedback": "",
                    "human_verdict": "",
                    "adjustment_hint": "",
                }
            )

    out_df = pd.DataFrame(rows)
    out_csv = args.out_csv if args.out_csv is not None else (doe_root / "doe_hil_queue_v1.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"DOE root: {doe_root_value}")
    print(f"Wrote: {out_csv}")
    print(f"Rows: {len(out_df)} (top/mid/bottom = {int(args.per_bucket)} each)")


if __name__ == "__main__":
    main()
