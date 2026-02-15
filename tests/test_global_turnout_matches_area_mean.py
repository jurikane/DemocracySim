from __future__ import annotations

import numpy as np

from tests.factory import create_test_model


def test_global_turnout_equals_population_weighted_area_turnout() -> None:
    """Regression: global turnout uses resident-population weighting across stepped areas."""

    model, _ = create_test_model(seed=123, num_agents=50, num_colors=3, num_areas=2, max_steps=3)

    # Run a couple of steps so areas have turnout values.
    model.step()
    model.step()

    df_model = model.datacollector.get_model_vars_dataframe()
    assert "turnout" in df_model.columns

    global_turnout_series = df_model["turnout"].to_numpy(dtype=float)

    total_participants = float(sum(int(area.num_agents_participated_last) for area in model.areas))
    total_resident = float(sum(int(area.num_agents) for area in model.areas))
    expected = (100.0 * total_participants / total_resident) if total_resident > 0.0 else 0.0

    # The last recorded model turnout should match resident-population weighted turnout.
    assert np.isclose(float(global_turnout_series[-1]), expected)
