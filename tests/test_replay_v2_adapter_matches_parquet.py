import numpy as np
import pandas as pd
import pytest

from src.config.loader import load_config
from src.replay.replay_server import ReplayModel
from src.logging.output_schema_v2 import (
    validate_steps_df,
    validate_area_steps_df,
)
from scripts.run_headless import run_once


@pytest.fixture()
def v2_run_dir(tmp_path):
    """Create a tiny schema v2 run directory under tmp_path."""
    out_dir = tmp_path / "v2_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config("configs/test.yaml")

    # Ensure the run is small/fast for CI.
    # We'll clone the config so we don't mutate globals.
    cfg_for_run = cfg.model_copy(deep=True)
    if hasattr(cfg_for_run, "simulation"):
        setattr(cfg_for_run.simulation, "num_steps", int(getattr(cfg_for_run.simulation, "num_steps", 2) or 2))
        # Ensure grids are available if replay expects them.
        setattr(cfg_for_run.simulation, "store_grid", True)
        setattr(cfg_for_run.simulation, "grid_interval", 1)

    run_once(run_id=0, cfg=cfg_for_run, out_dir=out_dir)

    meta = out_dir / "meta.yaml"
    assert meta.exists(), "No meta.yaml produced by headless run (schema v2)."
    return out_dir


def _steps_from_replay(run_dir):
    """Load v2 run via ReplayModel and materialize all steps into DataFrames."""
    appcfg = load_config("configs/test.yaml")
    model = ReplayModel(appcfg=appcfg, run_dir=run_dir)

    for _ in range(len(model.data)):
        model.step()

    model_df = model.datacollector.get_model_vars_dataframe().copy()
    area_df = model.datacollector.get_agent_vars_dataframe().copy()

    assert not model_df.empty, "Replay adapter model vars dataframe is empty."
    assert not area_df.empty, "Replay adapter area vars dataframe is empty."

    return model_df, area_df


def test_replay_v2_adapter_matches_parquet(v2_run_dir):
    run_dir = v2_run_dir

    # Source of truth
    steps_path = run_dir / "steps.parquet"
    area_steps_path = run_dir / "area_steps.parquet"
    assert steps_path.exists(), "steps.parquet missing in v2 run dir."
    assert area_steps_path.exists(), "area_steps.parquet missing in v2 run dir."

    steps = pd.read_parquet(steps_path)
    area_steps = pd.read_parquet(area_steps_path)
    validate_steps_df(steps)
    validate_area_steps_df(area_steps)

    # Replay adapter
    model_df, area_df = _steps_from_replay(run_dir)

    # ---- model-level scalars ----
    for col in ["collective_assets", "turnout", "gini_index"]:
        assert col in steps.columns, f"{col} missing in steps.parquet"
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

    for col in ["turnout", "dist_to_reality", "gini_index", "area_color_distribution", "elected_color"]:
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
