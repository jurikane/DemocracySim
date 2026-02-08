import numpy as np

from tests.factory import create_test_model


def _extract_step_agent_ids(agent_df):
    """
    Mesa DataCollector agent vars dataframe can be either:
      - columns: Step, AgentID, <vars...>   (older)
      - MultiIndex index: (Step, AgentID)   (common)
    Return (steps, agent_ids) as arrays.
    """
    if "Step" in agent_df.columns and "AgentID" in agent_df.columns:
        steps = agent_df["Step"].to_numpy()
        agent_ids = agent_df["AgentID"].to_numpy()
        return steps, agent_ids

    # MultiIndex case
    idx = agent_df.index
    if hasattr(idx, "nlevels") and idx.nlevels >= 2:
        # Prefer named levels if present, otherwise assume (step, agent_id)
        names = list(idx.names)
        step_level = 0
        agent_level = 1
        if "Step" in names:
            step_level = names.index("Step")
        if "AgentID" in names:
            agent_level = names.index("AgentID")

        steps = idx.get_level_values(step_level).to_numpy()
        agent_ids = idx.get_level_values(agent_level).to_numpy()
        return steps, agent_ids

    raise AssertionError(
        "Unexpected agent vars dataframe shape: expected Step/AgentID columns or MultiIndex (Step, AgentID)."
    )


def test_live_datacollector_emits_snake_case_and_vectors():
    """Contract: live model DataCollector uses snake_case and vector-valued area series.

    This is the minimum needed for run.py visualization:
      - model vars: collective_assets, turnout, gini_index, color_0..color_{C-1}
      - area vars: area_color_distribution (list[float]), elected_color (list[int]),
                   dist_to_reality, gini_index

    We step the model a couple times and assert columns exist and are non-empty.
    """
    model, _cfg = create_test_model()

    # Step a couple times so elections happen and area elections have results.
    for _ in range(3):
        model.step()

    model_df = model.datacollector.get_model_vars_dataframe()
    assert model_df is not None and len(model_df) > 0

    # --- model vars ---
    required_model_cols = {"collective_assets", "turnout", "gini_index"}
    missing = required_model_cols - set(model_df.columns)
    assert not missing, f"Missing model reporter columns: {sorted(missing)}"

    # At least one non-zero value where it should be plausible.
    assert (model_df["collective_assets"] != 0).any(), "collective_assets is all zeros"

    # colors
    num_colors = int(model.num_colors)
    assert num_colors > 0
    for i in range(num_colors):
        col = f"color_{i}"
        assert col in model_df.columns, f"Missing model color column: {col}"
        # distributions should be in [0,1] and not all zero
        vals = model_df[col].to_numpy(dtype=float)
        assert np.all(vals >= 0.0) and np.all(vals <= 1.0)
        assert np.any(vals > 0.0), f"{col} is all zeros"

    # --- area vars (agent df) ---
    area_df = model.datacollector.get_agent_vars_dataframe()
    assert area_df is not None and len(area_df) > 0

    required_area_cols = {"turnout", "dist_to_reality", "gini_index",
                          "area_color_distribution", "elected_color"}
    missing = required_area_cols - set(area_df.columns)
    assert not missing, f"Missing area reporter columns: {sorted(missing)}"

    # Robustly extract agent ids (areas) regardless of Mesa DF format
    _steps, agent_ids = _extract_step_agent_ids(area_df)

    # Expected area ids = whatever the model considers its real areas
    expected_area_ids = [int(a.unique_id) for a in getattr(model, "areas", [])]
    assert expected_area_ids, "Model has no areas configured"

    observed_ids = sorted(
        {int(x) for x in agent_ids if x is not None and str(x) != "nan"})
    intersect = sorted(set(observed_ids).intersection(expected_area_ids))

    assert intersect, (
        "No area rows found in agent vars dataframe. "
        f"Observed agent_ids={observed_ids[:20]} expected_area_ids={expected_area_ids[:20]}"
    )

    one_area_id = intersect[0]

    # Select rows for that one area id (works for both DF shapes)
    if "AgentID" in area_df.columns:
        sub = area_df[
            area_df["AgentID"].astype("Int64") == one_area_id].sort_values(
            "Step")
    else:
        # MultiIndex selection
        sub = area_df.xs(one_area_id, level=1, drop_level=False).sort_index()

    vectors = sub["area_color_distribution"].dropna()
    assert len(vectors) > 0, "area_color_distribution is empty"
    v0 = vectors.iloc[0]
    assert isinstance(v0,
                      list), f"area_color_distribution must be list, got {type(v0)}"
    assert len(
        v0) == num_colors, f"area_color_distribution length {len(v0)} != num_colors {num_colors}"

    elected = sub["elected_color"].dropna()
    assert len(elected) > 0, "elected_color is empty"
    e0 = elected.iloc[0]
    assert isinstance(e0, list), f"elected_color must be list, got {type(e0)}"
    assert len(
        e0) == num_colors, f"elected_color length {len(e0)} != num_colors {num_colors}"
    # Numeric series should exist
    assert sub["dist_to_reality"].dropna().abs().sum() >= 0.0
