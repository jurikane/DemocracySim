from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.run_headless import run_once
from src.config.loader import load_config
from src.utils.metrics import gini_index_0_100


pytestmark = pytest.mark.phase1
matplotlib.use("Agg")


def _load_run_frames(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    steps = pd.read_parquet(run_dir / "steps.parquet").sort_values("step").reset_index(drop=True)
    area_steps = pd.read_parquet(run_dir / "area_steps.parquet").sort_values(["step", "area_id"]).reset_index(drop=True)
    agents = pd.read_parquet(run_dir / "agents.parquet").sort_values(["step", "agent_id"]).reset_index(drop=True)
    votes = pd.read_parquet(run_dir / "votes.parquet").sort_values(["step", "area_id", "agent_id"]).reset_index(drop=True)
    return steps, area_steps, agents, votes


def _analyze_run_dir(run_dir: Path) -> dict:
    """Thesis analysis smoke: derive core outputs from logged artifacts only."""
    steps, area_steps, agents, votes = _load_run_frames(run_dir)

    # --- schema/units sanity on thesis-critical series
    assert {"step", "turnout", "gini_index", "collective_assets"}.issubset(steps.columns)
    assert {"step", "area_id", "turnout", "participants", "dist_to_reality"}.issubset(area_steps.columns)

    turnout = steps["turnout"].to_numpy(dtype=float)
    gini = steps["gini_index"].to_numpy(dtype=float)
    assets = steps["collective_assets"].to_numpy(dtype=float)
    assert np.all(np.isfinite(turnout)) and np.all((turnout >= 0.0) & (turnout <= 100.0))
    # Schema contract stores gini in 0..100.
    assert np.all(np.isfinite(gini)) and np.all((gini >= 0.0) & (gini <= 100.0 + 1e-6))
    assert np.all(np.isfinite(assets))

    # --- cross-table consistency checks (silent wrongness guard)
    # global turnout must equal mean area turnout per step
    area_turnout_mean = area_steps.groupby("step", sort=True)["turnout"].mean().reset_index(drop=True)
    np.testing.assert_allclose(
        area_turnout_mean.to_numpy(dtype=float),
        steps["turnout"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-6,
    )

    # participants in area_steps must match votes row counts per (step, area_id)
    votes_counts = votes.groupby(["step", "area_id"], sort=True).size().rename("vote_rows").reset_index()
    area_participants = area_steps[["step", "area_id", "participants"]].copy()
    merged = area_participants.merge(votes_counts, on=["step", "area_id"], how="left")
    merged["vote_rows"] = merged["vote_rows"].fillna(0).astype(int)
    np.testing.assert_array_equal(
        merged["participants"].to_numpy(dtype=int),
        merged["vote_rows"].to_numpy(dtype=int),
    )

    # one agent row per agent per step
    n_steps = int(steps["step"].nunique())
    n_agents = int(agents["agent_id"].nunique())
    assert len(agents) == n_steps * n_agents

    # --- dissatisfaction-inequality pipeline lock
    assert {"step", "agent_id", "dissatisfaction_value"}.issubset(agents.columns)
    diss_by_step = (
        agents.groupby("step", sort=True)["dissatisfaction_value"]
        .apply(lambda s: float(gini_index_0_100(s.to_numpy(dtype=float))))
        .reset_index(drop=True)
    )
    assert len(diss_by_step) == n_steps
    assert np.all(np.isfinite(diss_by_step.to_numpy(dtype=float)))
    assert np.all((diss_by_step.to_numpy(dtype=float) >= 0.0) & (diss_by_step.to_numpy(dtype=float) <= 100.0 + 1e-6))

    meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
    rule_idx = int(meta["run"]["rule_idx"])

    return {
        "run_dir": str(run_dir),
        "rule_idx": rule_idx,
        "n_steps": n_steps,
        "n_agents": n_agents,
        "mean_turnout": float(np.mean(turnout)),
        "final_turnout": float(turnout[-1]),
        "mean_gini": float(np.mean(gini)),
        "final_gini": float(gini[-1]),
        "mean_gini_dissatisfaction": float(np.mean(diss_by_step.to_numpy(dtype=float))),
        "final_gini_dissatisfaction": float(diss_by_step.to_numpy(dtype=float)[-1]),
        "final_collective_assets": float(assets[-1]),
        "mean_dist_to_reality": float(np.mean(area_steps["dist_to_reality"].to_numpy(dtype=float))),
    }


def _render_thesis_smoke_plot(run_dirs: list[Path], out_png: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for run_dir in run_dirs:
        steps = pd.read_parquet(run_dir / "steps.parquet").sort_values("step")
        meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
        label = f"rule_{int(meta['run']['rule_idx'])}"
        ax1.plot(steps["step"], steps["turnout"], label=label)
        ax2.plot(steps["step"], steps["gini_index"], label=label)

    ax1.set_ylabel("turnout [%]")
    ax2.set_ylabel("gini [0..100]")
    ax2.set_xlabel("step")
    ax1.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def test_thesis_analysis_pipeline_smoke_from_logged_artifacts_only(tmp_path: Path) -> None:
    """
    End-to-end thesis smoke test:
    1) run a small multi-rule batch,
    2) derive thesis-level summaries from parquet artifacts only,
    3) render a plot artifact from those same outputs.
    """
    cfg = load_config("toy.yaml").model_copy(deep=True)
    cfg.simulation.num_steps = 8
    cfg.simulation.store_grid = False
    cfg.simulation.base_seed = 9100

    batch_root = tmp_path / "batch"
    run_dirs: list[Path] = []
    for rule_idx in (0, 1, 2, 3):
        cfg_rule = cfg.model_copy(deep=True)
        cfg_rule.model.rule_idx = rule_idx
        out = batch_root / f"rule_{rule_idx}" / "run_0"
        run_once(run_id=0, cfg=cfg_rule, out_dir=out)
        run_dirs.append(out)

    # analysis pipeline from logged artifacts only
    rows = [_analyze_run_dir(rd) for rd in run_dirs]
    summary = pd.DataFrame(rows).sort_values("rule_idx").reset_index(drop=True)

    out_summary = tmp_path / "analysis_summary.csv"
    out_plot = tmp_path / "analysis_smoke.png"
    summary.to_csv(out_summary, index=False)
    _render_thesis_smoke_plot(run_dirs, out_plot)

    assert out_summary.exists() and out_summary.stat().st_size > 0
    assert out_plot.exists() and out_plot.stat().st_size > 0
    assert set(summary["rule_idx"].tolist()) == {0, 1, 2, 3}
    assert np.all(np.isfinite(summary["mean_turnout"].to_numpy(dtype=float)))
    assert np.all(np.isfinite(summary["final_gini"].to_numpy(dtype=float)))
    assert np.all(np.isfinite(summary["mean_gini_dissatisfaction"].to_numpy(dtype=float)))
    assert np.all(np.isfinite(summary["final_gini_dissatisfaction"].to_numpy(dtype=float)))
