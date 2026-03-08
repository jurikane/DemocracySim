from __future__ import annotations

import numpy as np

from tests.factory import create_test_model
from src.viz.visualization_elements import AgentLearningHistograms


def test_live_agent_learning_reporters_exist_and_finite_after_steps() -> None:
    """Minimal gate for Option 1 fast feedback.

    Requirements:
    - model_reporters include mean_p_participation and mean_altruism
    - values are finite after a couple steps
    - histogram element renders non-empty HTML after 1-2 steps
    """

    model, _cfg = create_test_model(seed=123, num_agents=30, num_colors=3, num_areas=2)

    # Step twice so elections and learning variables exist.
    model.step()
    model.step()

    model_df = model.datacollector.get_model_vars_dataframe()
    assert model_df is not None and len(model_df) >= 2

    for col in ["mean_p_participation", "mean_altruism"]:
        assert col in model_df.columns, f"Missing model reporter: {col}"
        vals = model_df[col].to_numpy(dtype=float)
        assert np.all(np.isfinite(vals)), f"Non-finite values in reporter {col}: {vals}"

    # Element should render a base64 image tag.
    el = AgentLearningHistograms()
    html = el.render(model)
    assert isinstance(html, str)
    assert "data:image/png;base64" in html
