from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_headless import run_once
from src.config.loader import load_config
from src.logging.run_logger import RunLoggerV2
from src.model_setup import make_model
from src.utils.metrics import gini_index_0_100


def _delta_l1_normalized(x: np.ndarray, y: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64))))


def _unique_rows(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr.reshape(0, 0)
    return np.unique(np.round(arr, decimals=12), axis=0).astype(np.float64)


def _compute_references(personal_dists: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    personal = np.asarray(personal_dists, dtype=np.float64)
    if personal.ndim != 2 or personal.shape[0] == 0:
        return None, None, None

    util = np.mean(personal, axis=0)
    util = util / float(util.sum())

    candidates = _unique_rows(np.vstack([personal, util.reshape(1, -1)]))
    if candidates.shape[0] == 0:
        return None, None, None

    dmat = 0.5 * np.sum(np.abs(candidates[:, None, :] - personal[None, :, :]), axis=2)
    means = np.mean(dmat, axis=1)
    worst = np.max(dmat, axis=1)
    gini = np.array([float(gini_index_0_100(row)) for row in dmat], dtype=np.float64)

    def _pick(primary: np.ndarray) -> np.ndarray:
        mask_primary = np.isclose(primary, float(np.min(primary)), rtol=0.0, atol=1e-12)
        sec = means.copy()
        sec_min = float(np.min(sec[mask_primary]))
        mask_secondary = np.isclose(sec, sec_min, rtol=0.0, atol=1e-12)
        chosen = candidates[mask_primary & mask_secondary]
        if chosen.shape[0] == 1:
            out = chosen[0]
        else:
            out = np.mean(chosen, axis=0)
        return out / float(np.sum(out))

    egal = _pick(gini)
    rawl = _pick(worst)
    return util, egal, rawl


def test_dist_to_ref_columns_exist_in_logged_tables(tmp_path: Path) -> None:
    cfg = load_config("toy.yaml").model_copy(deep=True)
    cfg.simulation.num_steps = 2
    cfg.simulation.store_grid = False
    cfg.simulation.base_seed = 9311

    run_dir = tmp_path / "run"
    run_once(0, cfg, out_dir=run_dir)

    steps = pd.read_parquet(run_dir / "steps.parquet")
    area_steps = pd.read_parquet(run_dir / "area_steps.parquet")

    for col in ("dist_to_ref_utilitarian", "dist_to_ref_egalitarian", "dist_to_ref_rawlsian"):
        assert col in steps.columns
        assert col in area_steps.columns


def test_dist_to_ref_values_follow_frozen_formula_area_and_global(tmp_path: Path) -> None:
    cfg = load_config("toy.yaml").model_copy(deep=True)
    cfg.simulation.num_steps = 1
    cfg.simulation.store_grid = False
    cfg.simulation.base_seed = 9312

    model = make_model(cfg.model)
    logger = RunLoggerV2(
        out_dir=tmp_path,
        run_seed=int(cfg.simulation.base_seed),
        rule_idx=int(cfg.model.rule_idx),
        num_steps=1,
        store_grid=False,
    )
    logger.attach_to_model(model)
    logger.begin_step(1)
    model.step()
    logger.log_step(step=1, model=model, grid_snapshot=None)
    logger.end_step()

    assert len(logger._steps_rows) == 1
    steps_row = logger._steps_rows[0]
    area_rows = {int(r["area_id"]): r for r in logger._area_steps_rows}

    # Area-level checks
    for area in model.areas:
        r = area_rows[int(area.unique_id)]
        personal = np.asarray([np.asarray(a.personal_opt_dist, dtype=np.float64) for a in area.agents], dtype=np.float64)
        util, egal, rawl = _compute_references(personal)
        area_color = np.asarray([float(r[f"area_color_{i}"]) for i in range(int(model.num_colors))], dtype=np.float64)

        expected = {
            "dist_to_ref_utilitarian": np.nan if util is None else _delta_l1_normalized(area_color, util),
            "dist_to_ref_egalitarian": np.nan if egal is None else _delta_l1_normalized(area_color, egal),
            "dist_to_ref_rawlsian": np.nan if rawl is None else _delta_l1_normalized(area_color, rawl),
        }
        for key, exp in expected.items():
            got = float(r[key])
            if np.isnan(exp):
                assert np.isnan(got)
            else:
                np.testing.assert_allclose(got, exp, rtol=0.0, atol=1e-6)

    # Global checks
    global_personal = np.asarray(
        [np.asarray(a.personal_opt_dist, dtype=np.float64) for a in model.voting_agents],
        dtype=np.float64,
    )
    g_util, g_egal, g_rawl = _compute_references(global_personal)
    global_color = np.asarray([float(steps_row[f"color_{i}"]) for i in range(int(model.num_colors))], dtype=np.float64)

    global_expected = {
        "dist_to_ref_utilitarian": np.nan if g_util is None else _delta_l1_normalized(global_color, g_util),
        "dist_to_ref_egalitarian": np.nan if g_egal is None else _delta_l1_normalized(global_color, g_egal),
        "dist_to_ref_rawlsian": np.nan if g_rawl is None else _delta_l1_normalized(global_color, g_rawl),
    }
    for key, exp in global_expected.items():
        got = float(steps_row[key])
        if np.isnan(exp):
            assert np.isnan(got)
        else:
            np.testing.assert_allclose(got, exp, rtol=0.0, atol=1e-6)


def test_dist_to_ref_nan_when_area_has_no_agents(tmp_path: Path) -> None:
    cfg = load_config("toy.yaml").model_copy(deep=True)
    cfg.model.num_agents = 1
    cfg.model.num_areas = 4
    cfg.model.height = 10
    cfg.model.width = 10
    cfg.model.av_area_height = 5
    cfg.model.av_area_width = 5
    cfg.model.area_size_variance = 0.0
    cfg.simulation.num_steps = 1
    cfg.simulation.store_grid = False
    cfg.simulation.base_seed = 9313

    run_dir = tmp_path / "run_nan"
    run_once(0, cfg, out_dir=run_dir)
    area_steps = pd.read_parquet(run_dir / "area_steps.parquet")

    empty_rows = area_steps[area_steps["eligible_voters"].astype(int) == 0]
    assert not empty_rows.empty, "Expected at least one area with zero agents in this setup."

    for col in ("dist_to_ref_utilitarian", "dist_to_ref_egalitarian", "dist_to_ref_rawlsian"):
        assert empty_rows[col].isna().all()

