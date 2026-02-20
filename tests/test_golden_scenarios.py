from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.run_headless import run_once
from src.config.loader import load_config


def _run_scenario(
    tmp_path: Path,
    *,
    label: str,
    bias: float,
    election_cost_rate: float,
    reward_rate_personal: float,
    num_steps: int = 12,
    run_seed_base: int = 7300,
) -> pd.DataFrame:
    cfg = load_config("toy.yaml").model_copy(deep=True)
    cfg.simulation.num_steps = int(num_steps)
    cfg.simulation.store_grid = False
    cfg.simulation.grid_interval = 1
    cfg.simulation.base_seed = int(run_seed_base)

    # Keep this moderate to reduce stochastic noise while remaining fast.
    cfg.model.num_agents = 80
    cfg.model.bias_toward_participation = float(bias)

    # Disable learning-driven drift so signatures stay stable and interpretable.
    cfg.model.participation_alpha = 0.0
    cfg.model.altruism_learning = False

    cfg.model.election_cost_rate = float(election_cost_rate)
    cfg.model.reward_rate_personal = float(reward_rate_personal)

    out_dir = tmp_path / label
    run_once(run_id=0, cfg=cfg, out_dir=out_dir)
    return pd.read_parquet(out_dir / "steps.parquet").sort_values("step").reset_index(drop=True)


def test_golden_scenarios_turnout_and_inequality_signatures() -> None:
    """P0 golden scenarios:
    1) turnout ordering under low/neutral/high participation bias
    2) inequality trajectory flat when economics are disabled
    3) inequality trajectory non-flat when participation cost is active
    """
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as td:
        tmp = Path(td)
        low = _run_scenario(
            tmp,
            label="low_bias",
            bias=-0.35,
            election_cost_rate=0.0,
            reward_rate_personal=0.0,
        )
        neutral = _run_scenario(
            tmp,
            label="neutral_bias",
            bias=0.0,
            election_cost_rate=0.0,
            reward_rate_personal=0.0,
        )
        high = _run_scenario(
            tmp,
            label="high_bias",
            bias=0.35,
            election_cost_rate=0.0,
            reward_rate_personal=0.0,
        )
        economy = _run_scenario(
            tmp,
            label="economy_cost_only",
            bias=0.0,
            election_cost_rate=0.05,
            reward_rate_personal=0.0,
        )

    # Qualitative turnout ordering (robust margins).
    m_low = float(low["turnout"].mean())
    m_neutral = float(neutral["turnout"].mean())
    m_high = float(high["turnout"].mean())
    assert m_low + 10.0 < m_neutral
    assert m_neutral + 10.0 < m_high

    # No-economy scenarios: collective assets and gini stay flat.
    for df in (low, neutral, high):
        assert float(df["collective_assets"].max() - df["collective_assets"].min()) == 0.0
        assert int(df["gini_index"].max() - df["gini_index"].min()) == 0

    # Economy-on scenario: inequality should move away from perfectly flat trajectory.
    assert int(economy["gini_index"].max() - economy["gini_index"].min()) > 0
