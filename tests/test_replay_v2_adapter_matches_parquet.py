import numpy as np
import pandas as pd
import yaml
import pytest

from src.config.loader import load_config
from src.replay.replay_server import ReplayModel
from src.logging.output_schema import (
    validate_steps_df,
    validate_area_steps_df,
)
from src.utils.metrics import gini_index_0_100
from scripts.run_headless import run_once


@pytest.fixture()
def replay_run_dir(tmp_path):
    """Create a tiny replay-compatible run directory under tmp_path."""
    out_dir = tmp_path / "v2_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config("configs/toy.yaml")

    # Ensure the run is small/fast for CI.
    # We'll clone the config so we don't mutate globals.
    cfg_for_run = cfg.model_copy(deep=True)
    if hasattr(cfg_for_run, "simulation"):
        # Ensure grids are available if replay expects them.
        setattr(cfg_for_run.simulation, "store_grid", True)
        setattr(cfg_for_run.simulation, "grid_interval", 1)

    run_once(run_id=0, cfg=cfg_for_run, out_dir=out_dir)

    meta = out_dir / "meta.yaml"
    assert meta.exists(), "No meta.yaml produced by headless run."
    return out_dir


def _steps_from_replay(run_dir):
    """Load run via ReplayModel and materialize all recorded steps into DataFrames."""
    appcfg = load_config("configs/toy.yaml")
    model = ReplayModel(appcfg=appcfg, run_dir=run_dir)

    # ReplayModel starts at step 0 (grid only). Materialize all recorded steps.
    for _ in range(len(model.data)):
        model.step()

    model_df = model.datacollector.get_model_vars_dataframe().copy()
    area_df = model.datacollector.get_agent_vars_dataframe().copy()

    assert not model_df.empty, "Replay adapter model vars dataframe is empty."
    assert not area_df.empty, "Replay adapter area vars dataframe is empty."

    return model_df, area_df


def test_replay_adapter_matches_parquet(replay_run_dir):
    run_dir = replay_run_dir

    # Source of truth
    steps_path = run_dir / "steps.parquet"
    area_steps_path = run_dir / "area_steps.parquet"
    agents_path = run_dir / "agents.parquet"
    assert steps_path.exists(), "steps.parquet missing in v2 run dir."
    assert area_steps_path.exists(), "area_steps.parquet missing in v2 run dir."
    assert agents_path.exists(), "agents.parquet missing in v2 run dir."

    steps = pd.read_parquet(steps_path)
    area_steps = pd.read_parquet(area_steps_path)
    agents = pd.read_parquet(agents_path)
    validate_steps_df(steps)
    validate_area_steps_df(area_steps)

    # Replay adapter
    model_df, area_df = _steps_from_replay(run_dir)

    # ---- model-level scalars ----
    for col in ["collective_assets", "turnout", "gini_index"]:
        assert col in steps.columns, f"{col} missing in steps.parquet"
        assert col in model_df.columns, f"{col} missing in replay model dataframe"
    for col in ["gini_dissatisfaction", "quality_distance"]:
        assert col in model_df.columns, f"{col} missing in replay model dataframe"
    for col in ["group_turnout", "group_mean_assets_share", "group_mean_dissatisfaction", "group_outcome_distance"]:
        assert col in model_df.columns, f"{col} missing in replay model dataframe"

    steps_sorted = steps.sort_values("step").reset_index(drop=True)
    model_sorted = model_df.reset_index(drop=True)

    np.testing.assert_array_equal(
        model_sorted["collective_assets"].to_numpy(),
        steps_sorted["collective_assets"].to_numpy(),
    )
    np.testing.assert_allclose(
        model_sorted["turnout"].to_numpy(dtype=float),
        steps_sorted["turnout"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        model_sorted["gini_index"].to_numpy(),
        steps_sorted["gini_index"].to_numpy(),
    )
    expected_gini_diss = (
        agents.groupby("step", sort=True)["dissatisfaction_value"]
        .apply(lambda s: float(gini_index_0_100(s.to_numpy(dtype=float))))
        .reindex(steps_sorted["step"].to_numpy(dtype=int), fill_value=np.nan)
        .to_numpy(dtype=float)
    )
    np.testing.assert_allclose(
        model_sorted["gini_dissatisfaction"].to_numpy(dtype=float),
        expected_gini_diss,
        rtol=0,
        atol=1e-6,
        equal_nan=True,
    )

    # Optional model color columns
    c = 0
    while f"color_{c}" in steps.columns:
        assert f"color_{c}" in model_df.columns, f"color_{c} missing in replay model dataframe"
        np.testing.assert_allclose(
            model_sorted[f"color_{c}"].to_numpy(dtype=float),
            steps_sorted[f"color_{c}"].to_numpy(dtype=float),
            rtol=0,
            atol=1e-6,
        )
        c += 1

    # ---- area-level: compare one area ----
    assert "area_id" in area_steps.columns, "area_id missing in area_steps.parquet"
    area_ids = sorted(int(x) for x in area_steps["area_id"].unique().tolist() if int(x) != -1)
    assert area_ids, "No non-global areas in area_steps.parquet"
    area_id = area_ids[0]

    a_parq = area_steps[area_steps["area_id"].astype(int) == area_id].sort_values("step")

    for col in [
        "turnout",
        "quality_distance",
        "dist_to_reality",
        "puzzle_distance",
        "gini_index",
        "area_color_distribution",
        "elected_color",
    ]:
        assert col in area_df.columns, f"{col} missing in replay area dataframe"

    a_rep = area_df.xs(area_id, level=1).sort_index()

    np.testing.assert_allclose(
        a_rep["turnout"].to_numpy(dtype=float),
        a_parq["turnout"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        a_rep["dist_to_reality"].to_numpy(dtype=float),
        a_parq["dist_to_reality"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        a_rep["puzzle_distance"].to_numpy(dtype=float),
        a_parq["puzzle_distance"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-6,
        equal_nan=True,
    )
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8")) or {}
    quality_mode = str((meta.get("run") or {}).get("quality_target_mode", "reality"))
    quality_src = "puzzle_distance" if quality_mode == "puzzle" else "dist_to_reality"
    weighted_quality_by_step = (
        area_steps.assign(
            _weighted=area_steps[quality_src].to_numpy(dtype=float)
            * area_steps["eligible_voters"].to_numpy(dtype=float)
        )
        .groupby("step", sort=True)[["_weighted", "eligible_voters"]]
        .sum()
    )
    weighted_quality = (
        weighted_quality_by_step["_weighted"]
        / weighted_quality_by_step["eligible_voters"]
    ).reindex(
        steps_sorted["step"].to_numpy(dtype=int),
        fill_value=np.nan,
    ).to_numpy(dtype=float)
    np.testing.assert_allclose(
        model_sorted["quality_distance"].to_numpy(dtype=float),
        weighted_quality,
        rtol=0,
        atol=1e-6,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        a_rep["quality_distance"].to_numpy(dtype=float),
        a_parq[quality_src].to_numpy(dtype=float),
        rtol=0,
        atol=1e-6,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        a_rep["gini_index"].to_numpy(),
        a_parq["gini_index"].to_numpy(),
    )

    # Vectors reconstructed from parquet columns
    def _vec_cols(prefix: str) -> list[str]:
        cols = [clm for clm in a_parq.columns if clm.startswith(prefix + "_")]
        assert cols, f"No {prefix}_* columns found in area_steps.parquet"
        return sorted(cols, key=lambda s: int(s.rsplit("_", 1)[-1]))

    area_color_cols = _vec_cols("area_color")
    elected_color_cols = _vec_cols("elected_color")

    area_color_parq = a_parq[area_color_cols].to_numpy(dtype=float)
    elected_color_parq = a_parq[elected_color_cols].to_numpy(dtype=int)

    # Replay stores list-of-length-C per step
    area_color_rep = np.vstack(a_rep["area_color_distribution"].apply(lambda x: np.asarray(x, dtype=float)).to_list())
    elected_color_rep = np.vstack(a_rep["elected_color"].apply(lambda x: np.asarray(x, dtype=int)).to_list())

    np.testing.assert_allclose(area_color_rep, area_color_parq, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(elected_color_rep, elected_color_parq)
