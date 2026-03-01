from __future__ import annotations

import argparse
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any

import pandas as pd


DEFAULT_QUEUE = Path("data") / "simulation_output" / "doe_hil_validation_template_v1.csv"
_SEED_RE = re.compile(r"seed_(\d+)")
_DESIGN_RE = re.compile(r"design_(\d+)")
_RULE_RE = re.compile(r"rule_([^/]+)")


def _resolve_doe_root_path(raw: str) -> Path:
    p = Path(str(raw))
    if p.exists():
        return p
    candidate = Path("data") / "simulation_output" / str(raw)
    if candidate.exists():
        return candidate
    return p


def _safe_float(v: Any, ndigits: int = 3) -> str:
    try:
        f = float(v)
    except Exception:
        return "n/a"
    if pd.isna(f):
        return "n/a"
    return f"{f:.{ndigits}f}"


def _safe_int(v: Any) -> str:
    try:
        i = int(round(float(v)))
    except Exception:
        return "n/a"
    return str(i)


def _build_ai_interpretation(
    *,
    bucket: str,
    design_row: pd.Series | None,
    run_row: pd.Series | None,
) -> str:
    if design_row is None and run_row is None:
        return "No matching DOE design/run records were found for this queue row."

    parts: list[str] = []
    if design_row is not None:
        parts.append(
            "Design ranking signals: "
            f"pass_rate={_safe_float(design_row.get('pass_rate'), 3)}, "
            f"score_total={_safe_float(design_row.get('score_total'), 3)}, "
            f"quality_mean={_safe_float(design_row.get('quality_mean'), 3)}, "
            f"seed_robustness={_safe_float(design_row.get('seed_robustness'), 3)}, "
            f"discriminability={_safe_float(design_row.get('discriminability'), 3)}."
        )

    if run_row is not None:
        gate_cols = [c for c in run_row.index if str(c).startswith("gate_")]
        failed = [c.replace("gate_", "") for c in gate_cols if not bool(run_row.get(c, False))]
        passes_hard = bool(run_row.get("passes_hard_gates", False))
        parts.append(
            f"Run hard gates: {'PASS' if passes_hard else 'FAIL'}; "
            f"failed={', '.join(failed) if failed else 'none'}."
        )
        parts.append(
            "Run dynamics summary: "
            f"mean_turnout={_safe_float(run_row.get('mean_turnout'), 1)} +/- {_safe_float(run_row.get('turnout_std'), 1)}, "
            f"mean_dist={_safe_float(run_row.get('mean_dist'), 3)} +/- {_safe_float(run_row.get('dist_std'), 3)}, "
            f"winner_changes_post_burnin={_safe_int(run_row.get('winner_changes_post_burnin'))}, "
            f"roll3_div_max={_safe_float(run_row.get('roll3_group_turnout_range_max'), 3)}, "
            f"roll20_div_max={_safe_float(run_row.get('roll20_group_turnout_range_max'), 3)}, "
            f"winner_entropy_norm={_safe_float(run_row.get('winner_entropy_norm'), 3)}."
        )
        new_signal_parts = []
        if "lag1_participation_signal_turnout_response_corr" in run_row.index:
            new_signal_parts.append(
                "lag1_resp(total)="
                f"{_safe_float(run_row.get('lag1_participation_signal_turnout_response_corr'), 3)}"
            )
        if "lag1_participation_signal_group_component_turnout_response_corr" in run_row.index:
            new_signal_parts.append(
                "lag1_resp(group)="
                f"{_safe_float(run_row.get('lag1_participation_signal_group_component_turnout_response_corr'), 3)}"
            )
        if "lag1_participation_signal_fee_component_turnout_response_corr" in run_row.index:
            new_signal_parts.append(
                "lag1_resp(fee)="
                f"{_safe_float(run_row.get('lag1_participation_signal_fee_component_turnout_response_corr'), 3)}"
            )
        if "puzzle_dominance_share_conflict" in run_row.index or "power_recovery_share_conflict" in run_row.index:
            new_signal_parts.append(
                "puzzle_conflict: "
                f"dom={_safe_float(run_row.get('puzzle_dominance_share_conflict'), 3)}, "
                f"recovery={_safe_float(run_row.get('power_recovery_share_conflict'), 3)}, "
                f"margin={_safe_float(run_row.get('puzzle_power_margin_mean_conflict'), 3)}"
            )
        if new_signal_parts:
            parts.append("Signal/puzzle diagnostics: " + "; ".join(new_signal_parts) + ".")
        reasons: list[str] = []
        if "no_lockin" in failed:
            reasons.append("early lock-in/static winner behavior")
        if "dist_activity" in failed:
            reasons.append("weak reality-distance activity")
        if "turnout_band" in failed:
            reasons.append("turnout outside target operating band")
        if "signal_present" in failed:
            reasons.append("weak temporal signal")
            if "group_divergence" in failed or "roll3_divergence" in failed or "roll20_divergence" in failed:
                reasons.append("insufficient cross-group turnout divergence")
            if "puzzle_anti_monopoly" in failed:
                reasons.append("puzzle dominance too monopolistic on conflict steps")
        if reasons:
            parts.append("Likely weak points: " + "; ".join(reasons) + ".")

    bucket_hint = {
        "top": "Bucket rationale: top bucket means this design ranks high under current selection objective.",
        "mid": "Bucket rationale: mid bucket means mixed/ambiguous quality under current objective.",
        "bottom": "Bucket rationale: bottom bucket means weak fit under current objective.",
    }.get(str(bucket), "Bucket rationale: run queued for visual calibration review.")
    parts.append(bucket_hint)
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare/execute HIL review commands from DOE visual validation queue."
    )
    parser.add_argument(
        "--queue-csv",
        type=Path,
        default=DEFAULT_QUEUE,
        help="Queue CSV (default: data/simulation_output/doe_hil_validation_template_v1.csv).",
    )
    parser.add_argument(
        "--bucket",
        choices=("all", "top", "mid", "bottom"),
        default="all",
        help="Filter by queue bucket.",
    )
    parser.add_argument(
        "--doe-root",
        type=str,
        default=None,
        help="Optional DOE root filter.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max rows to process (0 means all).",
    )
    parser.add_argument(
        "--pick-line",
        type=int,
        default=0,
        help="Pick exactly one CSV line number from listed rows (1-based, includes header line).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List selectable rows (CSV line, bucket, pass_rate, design_id, seed).",
    )
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help="Print `scripts.generate_summary` command lines for selected rows.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="List current selection and prompt for one CSV line number to execute.",
    )
    parser.add_argument(
        "--run-fast",
        action="store_true",
        help="Execute summary generation (`--mode fast --closed`) for each selected run.",
    )
    parser.add_argument(
        "--populate-ai",
        action="store_true",
        help="Populate `ai_interpretation` for selected rows and write back queue CSV.",
    )
    args = parser.parse_args()

    queue_csv = Path(args.queue_csv)
    if not queue_csv.exists():
        raise FileNotFoundError(f"Queue CSV not found: {queue_csv}")

    df = pd.read_csv(queue_csv)
    required = {"bucket", "representative_run_dir"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"Queue CSV missing required columns {sorted(required)}. Found: {list(df.columns)}"
        )
    for col in ("ai_interpretation", "human_feedback", "human_verdict", "adjustment_hint"):
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("").astype(str)

    sel = df.copy()
    sel["csv_line"] = (sel.index.astype(int) + 2).astype(int)
    if args.bucket != "all":
        sel = sel[sel["bucket"].astype(str) == str(args.bucket)]
    if args.doe_root is not None and "doe_root" in sel.columns:
        sel = sel[sel["doe_root"].astype(str) == str(args.doe_root)]
    sel = sel[sel["representative_run_dir"].astype(str).str.len() > 0]
    if int(args.limit) > 0:
        sel = sel.head(int(args.limit))

    seed_col = (
        sel["representative_run_dir"]
        .astype(str)
        .str.extract(_SEED_RE, expand=False)
        .fillna("")
    )
    sel = sel.assign(seed=seed_col)

    print(f"Queue: {queue_csv}")
    print(f"Selected rows: {len(sel)}")
    if len(sel) == 0:
        return

    if args.list or args.interactive:
        for _, row in sel.iterrows():
            print(
                f"line={int(row['csv_line'])} "
                f"bucket={row.get('bucket','')} "
                f"pass_rate={row.get('pass_rate','')} "
                f"design_id={row.get('design_id','')} "
                f"seed={row.get('seed','')}"
            )

    if args.interactive and int(args.pick_line) == 0:
        raw = input("Select CSV line number: ").strip()
        if raw == "":
            print("No line selected.")
            return
        args.pick_line = int(raw)

    if int(args.pick_line) > 0:
        sel = sel[sel["csv_line"].astype(int) == int(args.pick_line)]
        if len(sel) == 0:
            raise ValueError(f"--pick-line {args.pick_line} not found in current selection.")

    if args.populate_ai:
        cache: dict[str, tuple[pd.DataFrame | None, pd.DataFrame | None]] = {}
        updated = 0
        for idx, row in sel.iterrows():
            doe_root_raw = str(row.get("doe_root", "")).strip()
            if doe_root_raw not in cache:
                root = _resolve_doe_root_path(doe_root_raw)
                run_features_path = root / "doe_run_features.csv"
                design_scores_path = root / "doe_design_scores.csv"
                run_df = pd.read_csv(run_features_path) if run_features_path.exists() else None
                design_df = pd.read_csv(design_scores_path) if design_scores_path.exists() else None
                cache[doe_root_raw] = (run_df, design_df)
            run_df, design_df = cache[doe_root_raw]

            run_row: pd.Series | None = None
            design_row: pd.Series | None = None
            run_dir = str(row.get("representative_run_dir", "")).strip()
            seed = pd.to_numeric(seed_col.loc[idx], errors="coerce")
            design_id = pd.to_numeric(row.get("design_id"), errors="coerce")
            rule_match = _RULE_RE.search(run_dir)
            rule_name = rule_match.group(1) if rule_match else None
            design_match = _DESIGN_RE.search(run_dir)
            if pd.isna(design_id) and design_match:
                design_id = int(design_match.group(1))

            if run_df is not None:
                cand = run_df
                if run_dir:
                    exact = cand[cand["run_dir"].astype(str) == run_dir]
                    if len(exact) > 0:
                        cand = exact
                if pd.notna(design_id):
                    cand = cand[cand["design_id"].astype(int) == int(design_id)]
                if pd.notna(seed):
                    cand = cand[cand["seed"].astype(int) == int(seed)]
                if rule_name is not None and "rule_name" in cand.columns:
                    cand = cand[cand["rule_name"].astype(str) == str(rule_name)]
                if len(cand) > 0:
                    run_row = cand.iloc[0]

            if design_df is not None and pd.notna(design_id):
                d = design_df[design_df["design_id"].astype(int) == int(design_id)]
                if len(d) > 0:
                    design_row = d.iloc[0]

            text = _build_ai_interpretation(
                bucket=str(row.get("bucket", "")),
                design_row=design_row,
                run_row=run_row,
            )
            if str(df.at[idx, "ai_interpretation"]) != text:
                df.at[idx, "ai_interpretation"] = text
                updated += 1
        df.to_csv(queue_csv, index=False)
        print(f"Updated ai_interpretation for {updated} row(s).")

    commands: list[list[str]] = []
    for _, row in sel.iterrows():
        run_dir = str(row["representative_run_dir"]).strip()
        cmd = [
            sys.executable,
            "-m",
            "scripts.generate_summary",
            "--run-dir",
            run_dir,
            "--mode",
            "fast",
        ]
        commands.append(cmd)

    if args.print_commands:
        for i, cmd in enumerate(commands, start=1):
            quoted = " ".join(shlex.quote(c) for c in cmd)
            line_no = int(sel.iloc[i - 1]["csv_line"])
            print(f"[{i}] line={line_no} {quoted}")

    run_fast = bool(args.run_fast or args.interactive or int(args.pick_line) > 0)
    if run_fast:
        for i, cmd in enumerate(commands, start=1):
            line_no = int(sel.iloc[i - 1]["csv_line"])
            print(f"[RUN {i}/{len(commands)} line={line_no}] {' '.join(shlex.quote(c) for c in cmd)}")
            subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
