from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MANIFEST = Path("configs/thesis/final_run_manifest_v1.csv")
DEFAULT_OUT = Path("artifacts/thesis_analysis_v1/tables/free-rider-support.csv")

RULE_GROUPS = {
    "approval": "context",
    "borda": "canonical",
    "plurality": "reference",
    "random": "reference",
    "schulze": "canonical",
    "utilitarian": "canonical",
}


def normalize_rule_name(rule_name: str) -> str:
    raw = str(rule_name).strip().lower()
    return "plurality" if raw == "majority" else raw


def resolve_run_dir(out_dir: str, rule_name: str) -> Path:
    run_dir = Path(out_dir)
    if run_dir.exists():
        return run_dir
    if normalize_rule_name(rule_name) == "plurality":
        legacy_dir = Path(str(out_dir).replace("rule_plurality", "rule_majority"))
        if legacy_dir.exists():
            return legacy_dir
    raise FileNotFoundError(f"Run directory does not exist: {out_dir}")


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[mask]
    yv = yv[mask]
    if xv.size < 2 or np.allclose(xv, xv[0]) or np.allclose(yv, yv[0]):
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def percentile_rank(values: pd.Series) -> pd.Series:
    return values.rank(method="average", pct=True)


def high_low_delta(values: pd.DataFrame, column: str) -> float:
    q25 = float(values["abstention_rate"].quantile(0.25))
    q75 = float(values["abstention_rate"].quantile(0.75))
    low = values.loc[values["abstention_rate"] <= q25, column]
    high = values.loc[values["abstention_rate"] >= q75, column]
    if low.empty or high.empty:
        return float("nan")
    return float(high.mean() - low.mean())


def summarize_run_agents(agents: pd.DataFrame) -> pd.DataFrame:
    agents = agents.sort_values(["agent_id", "step"])
    final = agents.groupby("agent_id", sort=False).tail(1).copy()
    mean_dissatisfaction = (
        agents.groupby("agent_id", sort=False)["dissatisfaction_value"]
        .mean()
        .rename("mean_dissatisfaction")
    )
    abstention_rate = (
        1.0 - agents.groupby("agent_id", sort=False)["participating"].mean()
    ).rename("abstention_rate")

    summary = (
        final[
            ["agent_id", "assets", "dissatisfaction_value", "personality_group_idx"]
        ]
        .rename(columns={"dissatisfaction_value": "final_dissatisfaction"})
        .merge(mean_dissatisfaction, on="agent_id", how="inner")
        .merge(abstention_rate, on="agent_id", how="inner")
    )
    summary["assets_rank"] = percentile_rank(summary["assets"])
    summary["final_dissatisfaction_rank"] = percentile_rank(summary["final_dissatisfaction"])
    summary["mean_dissatisfaction_rank"] = percentile_rank(summary["mean_dissatisfaction"])
    return summary


def build_metric_row(
    *,
    scope: str,
    rule_name: str,
    personality_group_idx: int | None,
    rows: pd.DataFrame,
) -> dict[str, object]:
    return {
        "scope": scope,
        "rule_name": rule_name,
        "rule_group": RULE_GROUPS[rule_name],
        "personality_group_idx": personality_group_idx,
        "corr_abstention_assets_rank_mean": safe_corr(
            rows["abstention_rate"], rows["assets_rank"]
        ),
        "corr_abstention_final_dissatisfaction_rank_mean": safe_corr(
            rows["abstention_rate"], rows["final_dissatisfaction_rank"]
        ),
        "corr_abstention_mean_dissatisfaction_rank_mean": safe_corr(
            rows["abstention_rate"], rows["mean_dissatisfaction_rank"]
        ),
        "delta_high_minus_low_assets_rank_mean": high_low_delta(rows, "assets_rank"),
        "delta_high_minus_low_final_dissatisfaction_rank_mean": high_low_delta(
            rows, "final_dissatisfaction_rank"
        ),
        "delta_high_minus_low_mean_dissatisfaction_rank_mean": high_low_delta(
            rows, "mean_dissatisfaction_rank"
        ),
    }


def aggregate_metric_rows(metric_rows: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        "corr_abstention_assets_rank_mean",
        "corr_abstention_final_dissatisfaction_rank_mean",
        "corr_abstention_mean_dissatisfaction_rank_mean",
        "delta_high_minus_low_assets_rank_mean",
        "delta_high_minus_low_final_dissatisfaction_rank_mean",
        "delta_high_minus_low_mean_dissatisfaction_rank_mean",
    ]
    grouped = (
        metric_rows.groupby(["scope", "rule_name", "rule_group", "personality_group_idx"], dropna=False)[value_cols]
        .mean()
        .reset_index()
    )
    grouped["n_runs"] = (
        metric_rows.groupby(["scope", "rule_name", "rule_group", "personality_group_idx"], dropna=False)
        .size()
        .to_numpy()
    )
    return grouped


def build_support_table(manifest: pd.DataFrame) -> pd.DataFrame:
    all_agents_rows: list[dict[str, object]] = []
    within_group_rows: list[dict[str, object]] = []
    group_size_rows: list[dict[str, object]] = []

    columns = [
        "step",
        "agent_id",
        "participating",
        "assets",
        "dissatisfaction_value",
        "personality_group_idx",
    ]

    for record in manifest.itertuples(index=False):
        rule_name = normalize_rule_name(record.rule_name)
        run_dir = resolve_run_dir(record.out_dir, record.rule_name)
        agents = pd.read_parquet(run_dir / "agents.parquet", columns=columns)
        summary = summarize_run_agents(agents)
        summary["group_size"] = (
            summary.groupby("personality_group_idx")["agent_id"].transform("count").astype(float)
        )

        all_agents_rows.append(
            build_metric_row(
                scope="all_agents_by_rule",
                rule_name=rule_name,
                personality_group_idx=None,
                rows=summary,
            )
        )

        group_size_rows.append(
            {
                "scope": "group_size_assets_by_rule",
                "rule_name": rule_name,
                "rule_group": RULE_GROUPS[rule_name],
                "personality_group_idx": pd.NA,
                "corr_group_size_assets_mean": safe_corr(
                    summary["group_size"], summary["assets"]
                ),
                "corr_group_size_assets_rank_mean": safe_corr(
                    summary["group_size"], summary["assets_rank"]
                ),
            }
        )

        for personality_group_idx, group_rows in summary.groupby("personality_group_idx", sort=True):
            group_rows = group_rows.copy()
            group_rows["assets_rank"] = percentile_rank(group_rows["assets"])
            group_rows["final_dissatisfaction_rank"] = percentile_rank(
                group_rows["final_dissatisfaction"]
            )
            group_rows["mean_dissatisfaction_rank"] = percentile_rank(
                group_rows["mean_dissatisfaction"]
            )
            within_group_rows.append(
                build_metric_row(
                    scope="within_group_by_rule_and_group",
                    rule_name=rule_name,
                    personality_group_idx=int(personality_group_idx),
                    rows=group_rows,
                )
            )

    all_agents = aggregate_metric_rows(pd.DataFrame(all_agents_rows))
    within_group = aggregate_metric_rows(pd.DataFrame(within_group_rows))

    within_group_mean = (
        within_group.groupby(["rule_name", "rule_group"], as_index=False)[
            [
                "corr_abstention_assets_rank_mean",
                "corr_abstention_final_dissatisfaction_rank_mean",
                "corr_abstention_mean_dissatisfaction_rank_mean",
                "delta_high_minus_low_assets_rank_mean",
                "delta_high_minus_low_final_dissatisfaction_rank_mean",
                "delta_high_minus_low_mean_dissatisfaction_rank_mean",
            ]
        ]
        .mean()
    )
    within_group_mean.insert(0, "scope", "within_group_mean_by_rule")
    within_group_mean["personality_group_idx"] = pd.NA
    within_group_mean["n_runs"] = (
        within_group.groupby(["rule_name", "rule_group"]).size().to_numpy()
    )

    by_personality_group_mean = (
        within_group.groupby(["personality_group_idx"], as_index=False)[
            [
                "corr_abstention_assets_rank_mean",
                "corr_abstention_final_dissatisfaction_rank_mean",
                "corr_abstention_mean_dissatisfaction_rank_mean",
                "delta_high_minus_low_assets_rank_mean",
                "delta_high_minus_low_final_dissatisfaction_rank_mean",
                "delta_high_minus_low_mean_dissatisfaction_rank_mean",
            ]
        ]
        .mean()
    )
    by_personality_group_mean.insert(0, "scope", "within_group_mean_by_personality_group")
    by_personality_group_mean.insert(1, "rule_name", "all_rules")
    by_personality_group_mean.insert(2, "rule_group", "all")
    by_personality_group_mean["n_runs"] = within_group.groupby(["personality_group_idx"]).size().to_numpy()

    combined = pd.concat(
        [all_agents, within_group_mean, by_personality_group_mean, within_group],
        ignore_index=True,
    )
    group_size = (
        pd.DataFrame(group_size_rows)
        .groupby(["scope", "rule_name", "rule_group", "personality_group_idx"], dropna=False)[
            ["corr_group_size_assets_mean", "corr_group_size_assets_rank_mean"]
        ]
        .mean()
        .reset_index()
    )
    group_size["n_runs"] = (
        pd.DataFrame(group_size_rows)
        .groupby(["scope", "rule_name", "rule_group", "personality_group_idx"], dropna=False)
        .size()
        .to_numpy()
    )
    combined = combined.merge(
        group_size,
        on=["scope", "rule_name", "rule_group", "personality_group_idx", "n_runs"],
        how="outer",
    )
    combined = combined[
        [
            "scope",
            "rule_name",
            "rule_group",
            "personality_group_idx",
            "n_runs",
            "corr_abstention_assets_rank_mean",
            "corr_abstention_final_dissatisfaction_rank_mean",
            "corr_abstention_mean_dissatisfaction_rank_mean",
            "delta_high_minus_low_assets_rank_mean",
            "delta_high_minus_low_final_dissatisfaction_rank_mean",
            "delta_high_minus_low_mean_dissatisfaction_rank_mean",
            "corr_group_size_assets_mean",
            "corr_group_size_assets_rank_mean",
        ]
    ]
    return combined.sort_values(
        ["scope", "rule_name", "personality_group_idx"], kind="stable"
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a standalone support CSV for aggregate and within-group free-rider checks."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    support = build_support_table(manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    support.to_csv(args.out, index=False)
    print(args.out.as_posix())


if __name__ == "__main__":
    main()
