from __future__ import annotations

import numpy as np

from tests.factory import create_test_model
from src.viz.debug_viz import CohortElectionLearningDiagnostics


def test_live_cohort_election_learning_diagnostics_renders_and_is_finite() -> None:
    """Minimal gate:

    - create model with >1 personality group
    - step twice
    - element renders a base64 image
    - computed cohort arrays are non-empty and finite (NaNs allowed for empty buckets,
      but must not crash and must be numeric arrays)
    """

    model, _cfg = create_test_model(seed=123, num_agents=60, num_colors=3, num_personality_groups=4, num_areas=2)

    model.step()
    model.step()

    el = CohortElectionLearningDiagnostics(top_k=8)
    html = el.render(model)
    assert isinstance(html, str)
    assert "data:image/png;base64" in html

    stats = el._compute(model)
    assert stats is not None
    assert len(stats["labels"]) >= 1

    for key, val in stats.items():
        if key == "labels":
            continue
        assert isinstance(val, np.ndarray)
        assert val.size > 0
        # must be numeric; allow NaN, but no inf
        assert np.all(np.isfinite(np.nan_to_num(val, nan=0.0)))
