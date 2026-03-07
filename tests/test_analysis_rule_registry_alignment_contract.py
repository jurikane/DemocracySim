from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.doe_scoring import _current_rule_power_ordering_for_run
from src.analysis.summary_tooling import _compute_area_power_direction_orderings


pytestmark = pytest.mark.phase1


def test_summary_power_direction_excludes_random_and_keeps_schulze() -> None:
    area_group_series = pd.DataFrame(
        {
            "step": [1, 1],
            "group_idx": [0, 1],
            "residents": [6, 4],
        }
    )
    personality_groups = np.asarray(
        [
            [0, 1, 2],
            [1, 2, 0],
        ],
        dtype=np.int64,
    )
    power_dirs = _compute_area_power_direction_orderings(
        area_group_series=area_group_series,
        personality_groups=personality_groups,
        num_colors=3,
        meta={"run": {"run_seed": 17, "distance_impl_name": "spearman_fr_order"}},
    )
    assert [int(x["rule_idx"]) for x in power_dirs] == [0, 1, 2, 3, 4]
    assert [str(x["rule_name"]) for x in power_dirs] == [
        "Majority",
        "Approval",
        "Utilitarian",
        "Borda",
        "Schulze",
    ]


def test_doe_scoring_power_ordering_accepts_shifted_rule_indices() -> None:
    static = {
        "personality_group_info": {
            "personality_groups": [
                [0, 1, 2],
                [1, 2, 0],
            ]
        }
    }
    agents = pd.DataFrame(
        {
            "step": [1, 1, 1, 1, 1],
            "personality_group_idx": [0, 0, 1, 1, 1],
        }
    )
    for rule_idx in (4, 5):
        ordering = _current_rule_power_ordering_for_run(
            meta={
                "run": {
                    "rule_idx": int(rule_idx),
                    "run_seed": 19,
                    "distance_impl_name": "spearman_fr_order",
                }
            },
            static=static,
            agents=agents,
            num_colors=3,
        )
        assert ordering is not None
        arr = np.asarray(ordering, dtype=np.int64)
        assert arr.shape == (3,)
        assert set(arr.tolist()) == {0, 1, 2}
