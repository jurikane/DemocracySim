from __future__ import annotations

import json

import pandas as pd

from scripts.run_headless import run_once
from src.config.loader import load_config
from src.replay.replay_server import ReplayModel


def test_replay_static_includes_borders_and_voter_counts(tmp_path):
    """Replay static-artifact completeness regression.

    New runs must write static artifacts needed for replay + analysis:
    - static.json exists and declares schema v3
    - static_cell_areas.parquet maps every grid cell to at least one area
    - static_cell_agents.parquet includes area_id per agent for area membership recovery

    This test generates a tiny run to avoid depending on checked-in sample data.
    """
    conf = load_config('configs/toy.yaml')
    sim_cfg = conf.simulation
    try:
        sim_cfg = sim_cfg.model_copy(deep=True)
        sim_cfg.num_steps = 1
        sim_cfg.grid_interval = 1
        sim_cfg.store_grid = True
    except Exception:
        pass

    run_dir = tmp_path / 'run_0'
    run_dir.mkdir(parents=True, exist_ok=True)

    run_once(0, conf, out_dir=run_dir)

    static = json.loads((run_dir / 'static.json').read_text())
    schema = static.get('schema', {})
    assert schema.get('name') == 'output_schema_v3'
    assert int(schema.get('version', 0) or 0) == 3

    cell_areas_path = run_dir / "static_cell_areas.parquet"
    assert cell_areas_path.exists()
    area_df = pd.read_parquet(cell_areas_path)
    assert {"x", "y", "area_id"}.issubset(area_df.columns)
    n_cells = int(static["height"]) * int(static["width"])
    assert int(len(area_df[["x", "y"]].drop_duplicates())) == n_cells

    cell_agents_path = run_dir / "static_cell_agents.parquet"
    assert cell_agents_path.exists()
    agents_df = pd.read_parquet(cell_agents_path)
    assert {"x", "y", "area_id", "agent_id", "personality_group_idx"}.issubset(agents_df.columns)

    # Replay must derive border flags from cell_areas without a border artifact file.
    m = ReplayModel(appcfg=conf, run_dir=run_dir)
    assert any(bool(c.is_border_cell) for c in m.color_cells)

    # Hard cut: removed overlay artifacts are no longer written.
    assert not (run_dir / "cell_borders.parquet").exists()
    assert not (run_dir / "area_agents.parquet").exists()
    assert not (run_dir / "area_borders.npy").exists()
    assert not (run_dir / "agents_per_cell.npy").exists()
    assert not (run_dir / "agent_strings_per_cell.npy").exists()
    assert not (run_dir / "area_strings_per_cell.npy").exists()
