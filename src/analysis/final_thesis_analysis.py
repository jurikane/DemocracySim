from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import csv
import hashlib
import json
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.summary_io import _load_required_run_meta_static
from src.analysis.summary_series import _build_global_series
from src.analysis.thesis_endpoints import step_volatility_l1_normalized, time_mean, time_mean_last_frac


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FINAL_RUN_ROOT = Path("data/simulation_output/thesis_final_runs_v1")
DEFAULT_FINAL_MANIFEST = Path("configs/thesis/final_run_manifest_v1.csv")
DEFAULT_FREEZE_PROVENANCE = Path("configs/thesis/freeze_provenance_v1.json")
DEFAULT_PROTOCOL = Path("docs/internal/operations/thesis_analysis_protocol_v1.md")
DEFAULT_PACKAGE_OUT_DIR = Path("artifacts/thesis_analysis_v1")

DEFAULT_ANALYSIS_SEED = 20260311
DEFAULT_PERMUTATION_DRAWS = 100000
DEFAULT_BOOTSTRAP_REPS = 10000

PACKAGE_VERSION = "thesis_analysis_v1"
RUN_LEVEL_FILENAME = "run_level_endpoint_summary.csv"
RULE_STEP_FILENAME = "rule_step_primary_summary.csv"
CANONICAL_EFFECTS_FILENAME = "canonical_pairwise_effects.csv"
REFERENCE_EFFECTS_FILENAME = "reference_pairwise_effects.csv"
ROBUSTNESS_FILENAME = "robustness_alternative_readouts.csv"
F1_FILENAME = "F1_primary_metric_trajectories"
F2_FILENAME = "F2_paired_endpoint_comparisons"
F3_FILENAME = "F3_canonical_confirmatory_effect_forest"
F4_FILENAME = "F4_reference_family_effect_panel"

KNOWN_CAVEATS = [
    {
        "id": "seed_level_config_used",
        "summary": "config_used.yaml is stored at seed level rather than inside run_0.",
    },
    {
        "id": "stale_params_json_rule_idx",
        "summary": "params_json.rule_idx is stale auxiliary metadata in many manifest rows; manifest rule fields and meta.yaml remain authoritative.",
    },
    {
        "id": "asset_scale_explosive_but_finite",
        "summary": "collective asset magnitudes can become very large without producing numeric corruption.",
    },
]


@dataclass(frozen=True)
class AnalysisSourcePaths:
    run_root: Path
    manifest: Path
    freeze_provenance: Path
    protocol: Path


@dataclass(frozen=True)
class AnalysisOutputPaths:
    out_dir: Path
    derived_dir: Path
    tables_dir: Path
    figures_dir: Path
    readme_path: Path
    provenance_json_path: Path
    run_level_csv_path: Path
    rule_step_csv_path: Path


@dataclass(frozen=True)
class AnalysisSettings:
    analysis_seed: int = DEFAULT_ANALYSIS_SEED
    permutation_draws: int = DEFAULT_PERMUTATION_DRAWS
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS


def resolve_analysis_layout(
    *,
    run_root: Path | str = DEFAULT_FINAL_RUN_ROOT,
    manifest: Path | str = DEFAULT_FINAL_MANIFEST,
    freeze_provenance: Path | str = DEFAULT_FREEZE_PROVENANCE,
    protocol: Path | str = DEFAULT_PROTOCOL,
    out_dir: Path | str = DEFAULT_PACKAGE_OUT_DIR,
) -> tuple[AnalysisSourcePaths, AnalysisOutputPaths]:
    source = AnalysisSourcePaths(
        run_root=Path(run_root),
        manifest=Path(manifest),
        freeze_provenance=Path(freeze_provenance),
        protocol=Path(protocol),
    )
    missing = [str(path) for path in (source.run_root, source.manifest, source.freeze_provenance, source.protocol) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required analysis source path(s): {', '.join(missing)}")

    output = AnalysisOutputPaths(
        out_dir=Path(out_dir),
        derived_dir=Path(out_dir) / "derived",
        tables_dir=Path(out_dir) / "tables",
        figures_dir=Path(out_dir) / "figures",
        readme_path=Path(out_dir) / "README.md",
        provenance_json_path=Path(out_dir) / "analysis_provenance.json",
        run_level_csv_path=Path(out_dir) / "derived" / RUN_LEVEL_FILENAME,
        rule_step_csv_path=Path(out_dir) / "derived" / RULE_STEP_FILENAME,
    )
    return source, output


def build_initial_analysis_provenance(
    *,
    source: AnalysisSourcePaths,
    settings: AnalysisSettings,
) -> dict[str, Any]:
    freeze = json.loads(source.freeze_provenance.read_text(encoding="utf-8"))
    expected_runs_total = int(_freeze_expected_runs_total(freeze=freeze, manifest=source.manifest))
    rule_counts_expected = _load_rule_counts_expected(manifest=source.manifest)
    return {
        "version": PACKAGE_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_root": source.run_root.as_posix(),
        "source_manifest": source.manifest.as_posix(),
        "freeze_provenance": source.freeze_provenance.as_posix(),
        "protocol_path": source.protocol.as_posix(),
        "git_head": _git_head(),
        "analysis_seed": int(settings.analysis_seed),
        "permutation_draws": int(settings.permutation_draws),
        "bootstrap_reps": int(settings.bootstrap_reps),
        "expected_runs_total": expected_runs_total,
        "rule_counts_expected": rule_counts_expected,
        "known_caveats": KNOWN_CAVEATS,
    }


def write_analysis_scaffold(
    *,
    source: AnalysisSourcePaths,
    output: AnalysisOutputPaths,
    settings: AnalysisSettings,
) -> dict[str, Any]:
    for path in (output.out_dir, output.derived_dir, output.tables_dir, output.figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    output.readme_path.write_text(_build_package_readme(), encoding="utf-8")
    provenance = build_initial_analysis_provenance(source=source, settings=settings)
    output.provenance_json_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def build_base_analysis_outputs(
    *,
    source: AnalysisSourcePaths,
    output: AnalysisOutputPaths,
    settings: AnalysisSettings,
) -> dict[str, Any]:
    provenance = write_analysis_scaffold(source=source, output=output, settings=settings)

    manifest_rows = _load_manifest_rows(source.manifest)
    run_level_rows: list[dict[str, Any]] = []
    step_level_frames: list[pd.DataFrame] = []

    for row in manifest_rows:
        run_row, step_frame = _derive_run_outputs_for_manifest_row(row=row)
        run_level_rows.append(run_row)
        step_level_frames.append(step_frame)

    run_level_df = pd.DataFrame(run_level_rows).sort_values(["rule_idx", "seed"]).reset_index(drop=True)
    rule_step_df = _aggregate_rule_step_summary(step_level_frames=step_level_frames)

    run_level_df.to_csv(output.run_level_csv_path, index=False)
    rule_step_df.to_csv(output.rule_step_csv_path, index=False)

    provenance.update(
        {
            "realized_runs_total": int(len(run_level_df)),
            "rule_counts_realized": _rule_counts_from_run_level(run_level_df),
            "generated_artifacts": [
                output.run_level_csv_path.as_posix(),
                output.rule_step_csv_path.as_posix(),
            ],
        }
    )
    output.provenance_json_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def build_statistical_outputs(
    *,
    source: AnalysisSourcePaths,
    output: AnalysisOutputPaths,
    settings: AnalysisSettings,
) -> dict[str, Any]:
    provenance = build_base_analysis_outputs(source=source, output=output, settings=settings)

    run_level_df = pd.read_csv(output.run_level_csv_path)
    rule_step_df = pd.read_csv(output.rule_step_csv_path)

    canonical_df = _build_canonical_effects(run_level_df=run_level_df, settings=settings)
    reference_df = _build_reference_effects(run_level_df=run_level_df, settings=settings)
    robustness_df = _build_robustness_readouts(run_level_df=run_level_df, canonical_df=canonical_df)
    derived_paths = _derived_chunk3_paths(output)
    table_paths = _table_chunk3_paths(output)

    canonical_df.to_csv(derived_paths["canonical_pairwise_effects"], index=False)
    reference_df.to_csv(derived_paths["reference_pairwise_effects"], index=False)
    robustness_df.to_csv(derived_paths["robustness_alternative_readouts"], index=False)

    _build_table_t1(source=source, output=output, run_level_df=run_level_df, rule_step_df=rule_step_df).to_csv(
        table_paths["T1"], index=False
    )
    _build_table_t2().to_csv(table_paths["T2"], index=False)
    _build_table_t3(run_level_df=run_level_df).to_csv(table_paths["T3"], index=False)
    canonical_df.to_csv(table_paths["T4"], index=False)
    reference_df.to_csv(table_paths["T5"], index=False)
    robustness_df.to_csv(table_paths["T6"], index=False)

    generated = [
        output.run_level_csv_path.as_posix(),
        output.rule_step_csv_path.as_posix(),
        *(path.as_posix() for path in derived_paths.values()),
        *(path.as_posix() for path in table_paths.values()),
    ]
    provenance.update(
        {
            "generated_artifacts": generated,
            "realized_runs_total": int(len(run_level_df)),
            "rule_counts_realized": _rule_counts_from_run_level(run_level_df),
        }
    )
    output.provenance_json_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def build_full_analysis_package(
    *,
    source: AnalysisSourcePaths,
    output: AnalysisOutputPaths,
    settings: AnalysisSettings,
) -> dict[str, Any]:
    provenance = build_statistical_outputs(source=source, output=output, settings=settings)

    run_level_df = pd.read_csv(output.run_level_csv_path)
    step_df = pd.read_csv(output.rule_step_csv_path)
    derived_paths = _derived_chunk3_paths(output)
    canonical_df = pd.read_csv(derived_paths["canonical_pairwise_effects"])
    reference_df = pd.read_csv(derived_paths["reference_pairwise_effects"])

    figure_paths = _figure_chunk4_paths(output)
    _render_f1_primary_metric_trajectories(step_df=step_df, out_base=figure_paths["F1"])
    _render_f2_paired_endpoint_comparisons(run_level_df=run_level_df, out_base=figure_paths["F2"])
    _render_f3_canonical_effect_forest(canonical_df=canonical_df, out_base=figure_paths["F3"])
    _render_f4_reference_effect_panel(reference_df=reference_df, out_base=figure_paths["F4"])

    output.readme_path.write_text(_build_package_readme(include_deliverables=True), encoding="utf-8")

    generated = list(provenance.get("generated_artifacts", []))
    for path in figure_paths.values():
        generated.append(path.with_suffix(".png").as_posix())
        generated.append(path.with_suffix(".pdf").as_posix())
    provenance["generated_artifacts"] = generated
    output.provenance_json_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def _build_package_readme(*, include_deliverables: bool = False) -> str:
    lines = [
        "# Thesis Final Analysis Package",
        "",
        "This folder contains the tracked thesis-final analysis package derived from the frozen final run batch.",
        "",
        "Raw final-run directories are not tracked here.",
        "",
        "- `derived/` will hold machine-readable analysis products.",
        "- `tables/` will hold thesis-ready tables.",
        "- `figures/` will hold thesis-ready figures.",
        "- provenance is recorded in `analysis_provenance.json`.",
    ]
    if include_deliverables:
        lines.extend(
            [
                "",
                "Current tracked deliverables include:",
                "",
                "- `derived/run_level_endpoint_summary.csv`",
                "- `derived/rule_step_primary_summary.csv`",
                "- `derived/canonical_pairwise_effects.csv`",
                "- `derived/reference_pairwise_effects.csv`",
                "- `derived/robustness_alternative_readouts.csv`",
                "- `tables/T1_frozen_run_protocol_provenance.csv` through `tables/T6_robustness_summaries.csv`",
                "- `figures/F1_primary_metric_trajectories.(png|pdf)` through `figures/F4_reference_family_effect_panel.(png|pdf)`",
            ]
        )
    return "\n".join(lines) + "\n"


def _load_manifest_rows(manifest: Path) -> list[dict[str, Any]]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Manifest {manifest} is empty.")
    return rows


def _derive_run_outputs_for_manifest_row(*, row: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    run_dir = Path(str(row["out_dir"]))
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run directory referenced by manifest: {run_dir}")

    steps = pd.read_parquet(run_dir / "steps.parquet").sort_values("step").reset_index(drop=True)
    area_steps = pd.read_parquet(run_dir / "area_steps.parquet").sort_values(["step", "area_id"]).reset_index(drop=True)
    agents = pd.read_parquet(run_dir / "agents.parquet").sort_values(["step", "agent_id"]).reset_index(drop=True)
    votes = pd.read_parquet(run_dir / "votes.parquet").sort_values(["step", "area_id", "agent_id"]).reset_index(drop=True)
    meta, _static, num_colors = _load_required_run_meta_static(run_dir=run_dir)

    manifest_rule_idx = int(row["rule_idx"])
    executed_rule_idx = int(meta["run"]["rule_idx"])
    if manifest_rule_idx != executed_rule_idx:
        raise RuntimeError(
            f"Manifest/meta rule mismatch for {run_dir}: manifest={manifest_rule_idx}, meta={executed_rule_idx}"
        )

    quality_target_mode = str(meta["run"].get("quality_target_mode", "reality")).strip().lower()
    global_series = _build_global_series(
        steps=steps,
        area_steps=area_steps,
        agents=agents,
        votes=votes,
        num_colors=int(num_colors),
        refs_global=_empty_refs_global(),
        quality_target_mode=quality_target_mode,
    )

    rule_name = _normalize_rule_name(row.get("rule_name", ""))
    rule_group = _rule_group_for_rule_name(rule_name)

    run_level_row = {
        "design_id": int(row["design_id"]),
        "rule_idx": manifest_rule_idx,
        "rule_name": rule_name,
        "rule_group": rule_group,
        "seed": int(row["seed"]),
        "family_role": str(row.get("family_role", "")).strip(),
        "run_dir": run_dir.as_posix(),
        "quality_target_mode": quality_target_mode,
        "turnout_mean": _series_time_mean(global_series, "turnout"),
        "gini_assets_mean": _series_time_mean(global_series, "gini_assets"),
        "gini_dissatisfaction_mean": _series_time_mean(global_series, "gini_dissatisfaction"),
        "quality_distance_mean": _series_time_mean(global_series, "quality_distance"),
        "turnout_last_third_mean": _series_last_third_mean(global_series, "turnout"),
        "turnout_final": _series_final(global_series, "turnout"),
        "turnout_volatility": _series_volatility(global_series, "turnout", value_range=100.0),
        "gini_assets_last_third_mean": _series_last_third_mean(global_series, "gini_assets"),
        "gini_assets_final": _series_final(global_series, "gini_assets"),
        "gini_assets_volatility": _series_volatility(global_series, "gini_assets", value_range=100.0),
        "collective_assets_mean": _series_time_mean(steps, "collective_assets"),
        "collective_assets_last_third_mean": _series_last_third_mean(steps, "collective_assets"),
        "collective_assets_final": _series_final(steps, "collective_assets"),
        "gini_dissatisfaction_last_third_mean": _series_last_third_mean(global_series, "gini_dissatisfaction"),
        "gini_dissatisfaction_final": _series_final(global_series, "gini_dissatisfaction"),
        "gini_dissatisfaction_volatility": _series_volatility(global_series, "gini_dissatisfaction", value_range=100.0),
        "quality_distance_last_third_mean": _series_last_third_mean(global_series, "quality_distance"),
        "quality_distance_final": _series_final(global_series, "quality_distance"),
        "quality_distance_volatility": _series_volatility(global_series, "quality_distance", value_range=1.0),
        "mean_dissatisfaction_mean": _series_time_mean(global_series, "mean_dissatisfaction"),
        "mean_dissatisfaction_final": _series_final(global_series, "mean_dissatisfaction"),
        "diversity_entropy_mean": _series_time_mean(global_series, "diversity_first_choice_entropy"),
        "diversity_entropy_final": _series_final(global_series, "diversity_first_choice_entropy"),
    }

    step_frame = global_series.loc[
        :,
        ["step", "turnout", "gini_assets", "gini_dissatisfaction", "quality_distance"],
    ].copy()
    step_frame["collective_assets"] = steps["collective_assets"].to_numpy(dtype=float)
    step_frame["mean_dissatisfaction"] = steps["mean_dissatisfaction"].to_numpy(dtype=float)
    step_frame["rule_idx"] = manifest_rule_idx
    step_frame["rule_name"] = rule_name
    step_frame["rule_group"] = rule_group
    step_frame["seed"] = int(row["seed"])
    return run_level_row, step_frame


def _aggregate_rule_step_summary(*, step_level_frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not step_level_frames:
        raise RuntimeError("No step-level frames available for aggregation.")
    steps_all = pd.concat(step_level_frames, ignore_index=True)
    grouped = (
        steps_all.groupby(["rule_idx", "rule_name", "rule_group", "step"], sort=True)
        .agg(
            n_runs=("seed", "nunique"),
            turnout_mean=("turnout", "mean"),
            turnout_q25=("turnout", lambda s: float(s.quantile(0.25))),
            turnout_q75=("turnout", lambda s: float(s.quantile(0.75))),
            gini_assets_mean=("gini_assets", "mean"),
            gini_assets_q25=("gini_assets", lambda s: float(s.quantile(0.25))),
            gini_assets_q75=("gini_assets", lambda s: float(s.quantile(0.75))),
            collective_assets_mean=("collective_assets", "mean"),
            collective_assets_q25=("collective_assets", lambda s: float(s.quantile(0.25))),
            collective_assets_q75=("collective_assets", lambda s: float(s.quantile(0.75))),
            gini_dissatisfaction_mean=("gini_dissatisfaction", "mean"),
            gini_dissatisfaction_q25=("gini_dissatisfaction", lambda s: float(s.quantile(0.25))),
            gini_dissatisfaction_q75=("gini_dissatisfaction", lambda s: float(s.quantile(0.75))),
            mean_dissatisfaction_mean=("mean_dissatisfaction", "mean"),
            mean_dissatisfaction_q25=("mean_dissatisfaction", lambda s: float(s.quantile(0.25))),
            mean_dissatisfaction_q75=("mean_dissatisfaction", lambda s: float(s.quantile(0.75))),
            quality_distance_mean=("quality_distance", "mean"),
            quality_distance_q25=("quality_distance", lambda s: float(s.quantile(0.25))),
            quality_distance_q75=("quality_distance", lambda s: float(s.quantile(0.75))),
        )
        .reset_index()
        .sort_values(["rule_idx", "step"])
        .reset_index(drop=True)
    )
    return grouped


def _load_rule_counts_expected(*, manifest: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    with manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rule_name = _normalize_rule_name(row.get("rule_name", ""))
            if not rule_name:
                rule_name = f"rule_idx_{row.get('rule_idx', 'unknown')}"
            counts[rule_name] += 1
    return {name: int(counts[name]) for name in sorted(counts)}


def _freeze_expected_runs_total(*, freeze: dict[str, Any], manifest: Path) -> int:
    run_plan_counts = freeze.get("run_plan_counts")
    if isinstance(run_plan_counts, dict):
        raw = run_plan_counts.get("expected_runs_total")
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                pass
    with manifest.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _git_head() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "UNKNOWN"


def _empty_refs_global() -> dict[str, None]:
    return {
        "dist_to_ref_utilitarian": None,
        "dist_to_ref_nash": None,
        "dist_to_ref_rawlsian": None,
        "dist_to_ref_egalitarian": None,
        "dist_to_ref_egalitarian_lam025": None,
        "dist_to_ref_egalitarian_lam400": None,
    }


def _normalize_rule_name(value: Any) -> str:
    raw = str(value).strip().lower()
    # Backward compatibility for frozen pre-rename artifacts only.
    return "plurality" if raw == "majority" else raw


def _rule_group_for_rule_name(rule_name: str) -> str:
    if rule_name in {"utilitarian", "borda", "schulze"}:
        return "canonical"
    if rule_name in {"plurality", "random"}:
        return "reference"
    if rule_name == "approval":
        return "context"
    raise ValueError(f"Unsupported rule name in final analysis package: {rule_name!r}")


def _series_time_mean(df: pd.DataFrame, column: str) -> float:
    return float(time_mean(df[column].to_numpy(dtype=float)))


def _series_last_third_mean(df: pd.DataFrame, column: str) -> float:
    return float(time_mean_last_frac(df[column].to_numpy(dtype=float), frac=(1.0 / 3.0)))


def _series_final(df: pd.DataFrame, column: str) -> float:
    arr = df[column].to_numpy(dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr[-1])


def _series_volatility(df: pd.DataFrame, column: str, *, value_range: float) -> float:
    return float(step_volatility_l1_normalized(df[column].to_numpy(dtype=float), value_range=value_range))


def _rule_counts_from_run_level(df: pd.DataFrame) -> dict[str, int]:
    counts = df["rule_name"].value_counts().sort_index()
    return {str(name): int(count) for name, count in counts.items()}


ENDPOINT_SPECS = [
    {
        "endpoint": "turnout_mean",
        "endpoint_name": "Mean turnout",
        "source_series": "steps.turnout",
        "summary_operator": "time mean over 250 steps",
        "hypothesis_id": "H1",
        "last_third_col": "turnout_last_third_mean",
        "final_col": "turnout_final",
    },
    {
        "endpoint": "gini_assets_mean",
        "endpoint_name": "Mean asset inequality",
        "source_series": "steps.gini_index",
        "summary_operator": "time mean over 250 steps",
        "hypothesis_id": "H2",
        "last_third_col": "gini_assets_last_third_mean",
        "final_col": "gini_assets_final",
    },
    {
        "endpoint": "gini_dissatisfaction_mean",
        "endpoint_name": "Mean dissatisfaction inequality",
        "source_series": "agents.dissatisfaction_value -> gini_dissatisfaction",
        "summary_operator": "time mean over 250 steps",
        "hypothesis_id": "H3",
        "last_third_col": "gini_dissatisfaction_last_third_mean",
        "final_col": "gini_dissatisfaction_final",
    },
    {
        "endpoint": "quality_distance_mean",
        "endpoint_name": "Mean quality distance",
        "source_series": "area_steps.puzzle_distance (puzzle mode) / area_steps.dist_to_reality (reality mode)",
        "summary_operator": "time mean over 250 steps",
        "hypothesis_id": "H4",
        "last_third_col": "quality_distance_last_third_mean",
        "final_col": "quality_distance_final",
    },
]

CANONICAL_CONTRASTS = [
    ("utilitarian", "borda"),
    ("utilitarian", "schulze"),
    ("borda", "schulze"),
]

REFERENCE_CONTRASTS = [
    ("utilitarian", "plurality"),
    ("utilitarian", "random"),
    ("borda", "plurality"),
    ("borda", "random"),
    ("schulze", "plurality"),
    ("schulze", "random"),
]

def _build_canonical_effects(*, run_level_df: pd.DataFrame, settings: AnalysisSettings) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for endpoint_idx, spec in enumerate(ENDPOINT_SPECS):
        for contrast_idx, (left_rule, right_rule) in enumerate(CANONICAL_CONTRASTS):
            diffs = _paired_differences(run_level_df=run_level_df, left_rule=left_rule, right_rule=right_rule, value_col=spec["endpoint"])
            perm_seed = _analysis_subseed(settings.analysis_seed, "canonical", endpoint_idx, contrast_idx, "perm")
            boot_seed = _analysis_subseed(settings.analysis_seed, "canonical", endpoint_idx, contrast_idx, "boot")
            raw_p = _paired_sign_flip_pvalue(diffs=diffs, draws=settings.permutation_draws, seed=perm_seed)
            ci_low, ci_high = _bootstrap_mean_ci(diffs=diffs, reps=settings.bootstrap_reps, seed=boot_seed)
            rows.append(
                {
                    "endpoint": spec["endpoint"],
                    "endpoint_name": spec["endpoint_name"],
                    "contrast": f"{left_rule} - {right_rule}",
                    "left_rule": left_rule,
                    "right_rule": right_rule,
                    "n_pairs": int(diffs.size),
                    "paired_mean_difference": float(np.mean(diffs)),
                    "paired_median_difference": float(np.median(diffs)),
                    "ci_lower": ci_low,
                    "ci_upper": ci_high,
                    "raw_p": raw_p,
                }
            )
            raw_p_values.append(raw_p)

    adjusted = _holm_adjust(raw_p_values)
    for row, adj in zip(rows, adjusted, strict=True):
        row["holm_adjusted_p"] = float(adj)
        row["reject_0_05"] = bool(adj <= 0.05)
    return pd.DataFrame(rows)


def _build_reference_effects(*, run_level_df: pd.DataFrame, settings: AnalysisSettings) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for endpoint_idx, spec in enumerate(ENDPOINT_SPECS):
        for contrast_idx, (left_rule, right_rule) in enumerate(REFERENCE_CONTRASTS):
            diffs = _paired_differences(run_level_df=run_level_df, left_rule=left_rule, right_rule=right_rule, value_col=spec["endpoint"])
            perm_seed = _analysis_subseed(settings.analysis_seed, "reference", endpoint_idx, contrast_idx, "perm")
            boot_seed = _analysis_subseed(settings.analysis_seed, "reference", endpoint_idx, contrast_idx, "boot")
            raw_p = _paired_sign_flip_pvalue(diffs=diffs, draws=settings.permutation_draws, seed=perm_seed)
            ci_low, ci_high = _bootstrap_mean_ci(diffs=diffs, reps=settings.bootstrap_reps, seed=boot_seed)
            rows.append(
                {
                    "endpoint": spec["endpoint"],
                    "endpoint_name": spec["endpoint_name"],
                    "contrast": f"{left_rule} - {right_rule}",
                    "left_rule": left_rule,
                    "right_rule": right_rule,
                    "n_pairs": int(diffs.size),
                    "paired_mean_difference": float(np.mean(diffs)),
                    "paired_median_difference": float(np.median(diffs)),
                    "ci_lower": ci_low,
                    "ci_upper": ci_high,
                    "raw_p": raw_p,
                }
            )
            raw_p_values.append(raw_p)

    adjusted = _bh_adjust(raw_p_values)
    for row, adj in zip(rows, adjusted, strict=True):
        row["bh_q_value"] = float(adj)
    return pd.DataFrame(rows)


def _build_robustness_readouts(*, run_level_df: pd.DataFrame, canonical_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    canonical_lookup = {
        (str(row.endpoint), str(row.left_rule), str(row.right_rule)): float(row.paired_mean_difference)
        for row in canonical_df.itertuples(index=False)
    }
    spec_by_endpoint = {spec["endpoint"]: spec for spec in ENDPOINT_SPECS}
    for spec in ENDPOINT_SPECS:
        for left_rule, right_rule in CANONICAL_CONTRASTS:
            time_effect = canonical_lookup[(spec["endpoint"], left_rule, right_rule)]
            last_diffs = _paired_differences(
                run_level_df=run_level_df,
                left_rule=left_rule,
                right_rule=right_rule,
                value_col=spec["last_third_col"],
            )
            final_diffs = _paired_differences(
                run_level_df=run_level_df,
                left_rule=left_rule,
                right_rule=right_rule,
                value_col=spec["final_col"],
            )
            last_effect = float(np.mean(last_diffs))
            final_effect = float(np.mean(final_diffs))
            rows.append(
                {
                    "endpoint": spec["endpoint"],
                    "endpoint_name": spec["endpoint_name"],
                    "contrast": f"{left_rule} - {right_rule}",
                    "left_rule": left_rule,
                    "right_rule": right_rule,
                    "n_pairs": int(last_diffs.size),
                    "confirmatory_time_mean_effect": time_effect,
                    "last_third_mean_effect": last_effect,
                    "final_value_effect": final_effect,
                    "sign_match_last_third": bool(np.sign(time_effect) == np.sign(last_effect)),
                    "sign_match_final": bool(np.sign(time_effect) == np.sign(final_effect)),
                    "hypothesis_id": spec_by_endpoint[spec["endpoint"]]["hypothesis_id"],
                }
            )
    return pd.DataFrame(rows)


def _build_table_t1(
    *,
    source: AnalysisSourcePaths,
    output: AnalysisOutputPaths,
    run_level_df: pd.DataFrame,
    rule_step_df: pd.DataFrame,
) -> pd.DataFrame:
    freeze = json.loads(source.freeze_provenance.read_text(encoding="utf-8"))
    artifact_hashes = ((freeze.get("artifacts") or {}).get("sha256") or {})
    steps_per_run = int(rule_step_df["step"].nunique()) if not rule_step_df.empty else 0
    rows = [
        {"key": "git_head", "value": str(freeze.get("git_head", ""))},
        {"key": "doe_batch_id", "value": str(((freeze.get("source") or {}).get("doe_batch_id", "")))},
        {"key": "selected_design_id", "value": str(((freeze.get("source") or {}).get("selected_design_id", "")))},
        {"key": "config_hash", "value": str(artifact_hashes.get("configs/thesis/final_model_v1.yaml", ""))},
        {"key": "manifest_hash", "value": str(artifact_hashes.get("configs/thesis/final_run_manifest_v1.csv", ""))},
        {"key": "schema_version", "value": "schema-v2"},
        {"key": "expected_runs", "value": str(((freeze.get("run_plan_counts") or {}).get("expected_runs_total", "")))},
        {"key": "realized_runs", "value": str(int(len(run_level_df)))},
        {"key": "steps_per_run", "value": str(steps_per_run)},
    ]
    return pd.DataFrame(rows)


def _build_table_t2() -> pd.DataFrame:
    rows = []
    for spec in ENDPOINT_SPECS:
        rows.append(
            {
                "endpoint": spec["endpoint"],
                "endpoint_name": spec["endpoint_name"],
                "source_series": spec["source_series"],
                "summary_operator": spec["summary_operator"],
                "hypothesis_id": spec["hypothesis_id"],
                "canonical_test_pool": "yes",
                "reference_test_pool": "yes",
            }
        )
    return pd.DataFrame(rows)


def _build_table_t3(*, run_level_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    endpoint_cols = [spec["endpoint"] for spec in ENDPOINT_SPECS]
    descriptive_cols = [
        "collective_assets_mean",
        "mean_dissatisfaction_mean",
    ]
    ordered_rules = ["plurality", "approval", "utilitarian", "borda", "schulze", "random"]
    for rule_name in ordered_rules:
        block = run_level_df[run_level_df["rule_name"] == rule_name].copy()
        if block.empty:
            continue
        row: dict[str, Any] = {
            "rule_name": rule_name,
            "rule_group": str(block["rule_group"].iloc[0]),
            "n_runs": int(len(block)),
        }
        for endpoint in endpoint_cols:
            base = endpoint.replace("_mean", "")
            values = block[endpoint].to_numpy(dtype=float)
            row[f"{base}_mean"] = float(np.mean(values))
            row[f"{base}_sd"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        for col in descriptive_cols:
            base = col.replace("_mean", "")
            values = block[col].to_numpy(dtype=float)
            row[f"{base}_mean"] = float(np.mean(values))
            row[f"{base}_sd"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_differences(
    *,
    run_level_df: pd.DataFrame,
    left_rule: str,
    right_rule: str,
    value_col: str,
) -> np.ndarray:
    left = run_level_df.loc[run_level_df["rule_name"] == left_rule, ["seed", value_col]].rename(columns={value_col: "left_value"})
    right = run_level_df.loc[run_level_df["rule_name"] == right_rule, ["seed", value_col]].rename(columns={value_col: "right_value"})
    merged = left.merge(right, on="seed", how="inner").sort_values("seed").reset_index(drop=True)
    if merged.empty:
        raise RuntimeError(f"No matched seeds for contrast {left_rule} - {right_rule}.")
    diffs = merged["left_value"].to_numpy(dtype=float) - merged["right_value"].to_numpy(dtype=float)
    if not np.all(np.isfinite(diffs)):
        raise RuntimeError(f"Non-finite paired differences for contrast {left_rule} - {right_rule}, value {value_col}.")
    return diffs


def _paired_sign_flip_pvalue(*, diffs: np.ndarray, draws: int, seed: int) -> float:
    arr = np.asarray(diffs, dtype=float)
    if arr.size == 0:
        return float("nan")
    observed = abs(float(np.mean(arr)))
    rng = np.random.default_rng(int(seed))
    extreme = 0
    remaining = int(draws)
    chunk_size = 5000
    while remaining > 0:
        m = min(chunk_size, remaining)
        signs = rng.integers(0, 2, size=(m, arr.size), dtype=np.int8)
        signs = signs.astype(np.float64) * 2.0 - 1.0
        stats = np.abs((signs * arr[None, :]).mean(axis=1))
        extreme += int(np.sum(stats >= observed - 1e-15))
        remaining -= m
    return float((extreme + 1) / (int(draws) + 1))


def _bootstrap_mean_ci(*, diffs: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    arr = np.asarray(diffs, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(reps), dtype=np.float64)
    remaining = int(reps)
    start = 0
    chunk_size = 5000
    while remaining > 0:
        m = min(chunk_size, remaining)
        idx = rng.integers(0, arr.size, size=(m, arr.size))
        samples[start:start + m] = arr[idx].mean(axis=1)
        remaining -= m
        start += m
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def _bh_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    adjusted_sorted = (ranked * m) / (np.arange(1, m + 1))
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def _analysis_subseed(base_seed: int, *parts: Any) -> int:
    blob = "|".join([str(base_seed), *(str(part) for part in parts)]).encode("utf-8")
    digest = hashlib.sha256(blob).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**32 - 1)


def _derived_chunk3_paths(output: AnalysisOutputPaths) -> dict[str, Path]:
    return {
        "canonical_pairwise_effects": output.derived_dir / CANONICAL_EFFECTS_FILENAME,
        "reference_pairwise_effects": output.derived_dir / REFERENCE_EFFECTS_FILENAME,
        "robustness_alternative_readouts": output.derived_dir / ROBUSTNESS_FILENAME,
    }


def _table_chunk3_paths(output: AnalysisOutputPaths) -> dict[str, Path]:
    return {
        "T1": output.tables_dir / "T1_frozen_run_protocol_provenance.csv",
        "T2": output.tables_dir / "T2_endpoint_formulas_and_hypotheses.csv",
        "T3": output.tables_dir / "T3_endpoint_summaries_per_rule.csv",
        "T4": output.tables_dir / "T4_canonical_confirmatory_results.csv",
        "T5": output.tables_dir / "T5_reference_family_results.csv",
        "T6": output.tables_dir / "T6_robustness_summaries.csv",
    }


def _figure_chunk4_paths(output: AnalysisOutputPaths) -> dict[str, Path]:
    return {
        "F1": output.figures_dir / F1_FILENAME,
        "F2": output.figures_dir / F2_FILENAME,
        "F3": output.figures_dir / F3_FILENAME,
        "F4": output.figures_dir / F4_FILENAME,
    }


RULE_COLORS = {
    "utilitarian": "#1b9e77",
    "borda": "#d95f02",
    "schulze": "#7570b3",
    "plurality": "#6c6c6c",
    "random": "#1f1f1f",
    "approval": "#8da0cb",
}

ENDPOINT_LABELS = {
    "turnout_mean": "Mean turnout",
    "gini_assets_mean": "Asset inequality",
    "gini_dissatisfaction_mean": "Dissatisfaction inequality",
    "quality_distance_mean": "Quality distance",
}


def _render_f1_primary_metric_trajectories(*, step_df: pd.DataFrame, out_base: Path) -> None:
    metric_specs = [
        ("turnout", "Turnout", "turnout_mean", "turnout_q25", "turnout_q75"),
        ("gini_assets", "Asset inequality", "gini_assets_mean", "gini_assets_q25", "gini_assets_q75"),
        (
            "gini_dissatisfaction",
            "Dissatisfaction inequality",
            "gini_dissatisfaction_mean",
            "gini_dissatisfaction_q25",
            "gini_dissatisfaction_q75",
        ),
        ("quality_distance", "Quality distance", "quality_distance_mean", "quality_distance_q25", "quality_distance_q75"),
    ]
    canonical_rules = ["utilitarian", "borda", "schulze"]
    reference_rules = [("plurality", "--"), ("random", ":")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    for ax, (_metric_key, title, mean_col, q25_col, q75_col) in zip(axes, metric_specs, strict=True):
        for rule in canonical_rules:
            block = step_df[step_df["rule_name"] == rule].sort_values("step")
            ax.plot(block["step"], block[mean_col], color=RULE_COLORS[rule], linewidth=2.0, label=rule.title())
            ax.fill_between(
                block["step"].to_numpy(dtype=float),
                block[q25_col].to_numpy(dtype=float),
                block[q75_col].to_numpy(dtype=float),
                color=RULE_COLORS[rule],
                alpha=0.14,
                linewidth=0.0,
            )
        for rule, linestyle in reference_rules:
            block = step_df[step_df["rule_name"] == rule].sort_values("step")
            ax.plot(
                block["step"],
                block[mean_col],
                color=RULE_COLORS[rule],
                linewidth=1.8,
                linestyle=linestyle,
                label=rule.title(),
            )
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.grid(axis="y", alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    _save_figure(fig=fig, out_base=out_base)


def _render_f2_paired_endpoint_comparisons(*, run_level_df: pd.DataFrame, out_base: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    contrast_labels = [f"{a}\nvs {b}" for a, b in CANONICAL_CONTRASTS]
    box_colors = [RULE_COLORS["utilitarian"], RULE_COLORS["utilitarian"], RULE_COLORS["borda"]]
    for ax, spec in zip(axes, ENDPOINT_SPECS, strict=True):
        diff_sets = [
            _paired_differences(run_level_df=run_level_df, left_rule=a, right_rule=b, value_col=spec["endpoint"])
            for a, b in CANONICAL_CONTRASTS
        ]
        positions = np.arange(1, len(diff_sets) + 1)
        bp = ax.boxplot(
            diff_sets,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 1.2},
            whiskerprops={"color": "#555555", "linewidth": 1.0},
            capprops={"color": "#555555", "linewidth": 1.0},
        )
        for patch, color in zip(bp["boxes"], box_colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.18)
            patch.set_edgecolor(color)
            patch.set_linewidth(1.4)
        ax.scatter(positions, [float(np.mean(x)) for x in diff_sets], color="#111111", s=24, zorder=3)
        ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
        ax.set_xticks(positions, contrast_labels)
        ax.set_title(ENDPOINT_LABELS[spec["endpoint"]])
        ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    _save_figure(fig=fig, out_base=out_base)


def _render_f3_canonical_effect_forest(*, canonical_df: pd.DataFrame, out_base: Path) -> None:
    df = canonical_df.copy()
    df["endpoint_order"] = df["endpoint"].map({spec["endpoint"]: idx for idx, spec in enumerate(ENDPOINT_SPECS)})
    df["contrast_order"] = df["contrast"].map({f"{a} - {b}": idx for idx, (a, b) in enumerate(CANONICAL_CONTRASTS)})
    df = df.sort_values(["endpoint_order", "contrast_order"]).reset_index(drop=True)
    y_positions = np.arange(len(df))[::-1]
    endpoint_colors = {
        "turnout_mean": "#1b9e77",
        "gini_assets_mean": "#d95f02",
        "gini_dissatisfaction_mean": "#7570b3",
        "quality_distance_mean": "#e7298a",
    }
    endpoint_symbol_labels = {
        "turnout_mean": r"$\bar{T}_r$",
        "gini_assets_mean": r"$\bar{G}^{A}_r$",
        "gini_dissatisfaction_mean": r"$\bar{G}^{D}_r$",
        "quality_distance_mean": r"$\bar{Q}_r$",
    }

    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    span = float(df["ci_upper"].max() - df["ci_lower"].min())
    offset = 0.03 * (span if span > 0.0 else 1.0)
    for y, row in zip(y_positions, df.itertuples(index=False), strict=True):
        color = endpoint_colors[str(row.endpoint)]
        ax.errorbar(
            float(row.paired_mean_difference),
            y,
            xerr=[[float(row.paired_mean_difference) - float(row.ci_lower)], [float(row.ci_upper) - float(row.paired_mean_difference)]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.8,
            capsize=3,
            markersize=5,
        )
        ax.text(float(row.ci_upper) + offset, y, f"Holm p={float(row.holm_adjusted_p):.3g}", va="center", fontsize=8)
    labels = [f"{endpoint_symbol_labels[e]}: {c}" for e, c in zip(df["endpoint"], df["contrast"], strict=True)]
    ax.set_yticks(y_positions, labels)
    ax.set_xlabel("Paired mean difference")
    ax.set_title("Canonical confirmatory effects")
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    _save_figure(fig=fig, out_base=out_base)


def _render_f4_reference_effect_panel(*, reference_df: pd.DataFrame, out_base: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    row_order = [f"{a} - {b}" for a, b in REFERENCE_CONTRASTS]
    endpoint_order = [spec["endpoint"] for spec in ENDPOINT_SPECS]
    for ax, endpoint in zip(axes, endpoint_order, strict=True):
        block = reference_df[reference_df["endpoint"] == endpoint].copy()
        block["row_order"] = block["contrast"].map({name: idx for idx, name in enumerate(row_order)})
        block = block.sort_values("row_order").reset_index(drop=True)
        y_positions = np.arange(len(block))[::-1]
        ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1.0)
        span = float(block["ci_upper"].max() - block["ci_lower"].min())
        offset = 0.03 * (span if span > 0.0 else 1.0)
        for y, row in zip(y_positions, block.itertuples(index=False), strict=True):
            color = RULE_COLORS[str(row.left_rule)]
            ax.errorbar(
                float(row.paired_mean_difference),
                y,
                xerr=[[float(row.paired_mean_difference) - float(row.ci_lower)], [float(row.ci_upper) - float(row.paired_mean_difference)]],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.6,
                capsize=3,
                markersize=5,
            )
            ax.text(float(row.ci_upper) + offset, y, f"q={float(row.bh_q_value):.3g}", va="center", fontsize=8)
        ax.set_yticks(y_positions, block["contrast"].tolist())
        ax.set_title(ENDPOINT_LABELS[endpoint])
        ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    _save_figure(fig=fig, out_base=out_base)


def _save_figure(*, fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
