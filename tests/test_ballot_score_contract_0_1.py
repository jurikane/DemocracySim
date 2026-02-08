from __future__ import annotations

import numpy as np
import pytest

from tests.factory import create_test_model


@pytest.mark.parametrize("distance_idx", [0, 1])
def test_ballot_scores_in_unit_interval_for_distance_functions(distance_idx: int) -> None:
    """Ballot oppose-scores must be in [0,1] for each configured distance function.

    If this fails, fix the distance function normalization; do NOT clip/normalize ballots.
    """

    model, _cfg = create_test_model(seed=123, num_colors=4, num_agents=5, distance_idx=distance_idx)

    agent = model.voting_agents[0]
    assert agent is not None
    area = model.areas[0]
    assert area is not None

    # Match election pipeline semantics without depending on VoteAgent.update_known_cells.
    k = int(model.known_cells)
    agent.known_cells = list(area.cells[:k]) if len(area.cells) >= k else list(area.cells)

    scores = np.asarray(agent.vote(area), dtype=np.float32)

    assert scores.ndim == 1
    assert scores.shape[0] == model.options.shape[0]
    assert np.all(np.isfinite(scores))

    # Allow tiny epsilon due to float noise.
    assert float(scores.min()) >= -1e-6
    assert float(scores.max()) <= 1.0 + 1e-6
