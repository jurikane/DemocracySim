from __future__ import annotations

import numpy as np

from tests.factory import create_test_model


def test_global_turnout_equals_mean_of_area_turnouts_only() -> None:
    """Regression: global turnout must not include global_area.

    The scheduler steps only `model.areas` (not `model.global_area`), so the
    model-level turnout reporter should equal the mean of per-area turnouts.
    """

    model, _ = create_test_model(seed=123, num_agents=50, num_colors=3, num_areas=2, max_steps=3)

    # Run a couple of steps so areas have turnout values.
    model.step()
    model.step()

    df_model = model.datacollector.get_model_vars_dataframe()
    assert "turnout" in df_model.columns

    global_turnout_series = df_model["turnout"].to_numpy(dtype=float)

    # Mean of current per-area turnouts at each recorded step.
    area_turnouts = []
    for area in model.areas:
        area_turnouts.append(float(area.voter_turnout))
    expected = float(np.mean(area_turnouts))

    # The last recorded model turnout should match the current mean per-area turnout.
    assert np.isclose(float(global_turnout_series[-1]), expected)
