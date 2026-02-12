import numpy as np

from tests.factory import create_test_model


def _grid_distribution_from_cells(model) -> np.ndarray:
    counts = np.zeros(model.num_colors, dtype=np.float64)
    for c in model.color_cells:
        counts[int(c.color)] += 1.0
    return counts / float(len(model.color_cells))


def test_environment_pipeline_interaction_init_patch_then_mutate_then_global_distribution_exact():
    """
    Interaction test for Section E knobs:
    - heterogeneity influences the preset distribution used for initial colors
    - color_patches_steps + patch_power can change realized initial grid
    - mu drives mutation when turnout > 0
    - election_impact_on_mutation influences mutation sampling distribution
    - global_color_dst must match the realized grid throughout

    We stub out Area.step() to avoid coupling this environment test to the full
    voting/learning pipeline, while still exercising scheduler timing semantics:
    mutation happens at the start of step t+1.
    """
    model, _ = create_test_model(
        height=8, width=8,
        num_colors=4,
        num_agents=2,
        num_personality_groups=1,
        known_cells=1,
        num_areas=1,
        av_area_height=8, av_area_width=8,
        area_size_variance=0.0,
        heterogeneity=0.8,
        color_patches_steps=2,
        patch_power=2.0,
        mu=1.0,  # mutate all cells when an election occurred (turnout>0)
        election_impact_on_mutation=2.0,
        seed=123,
    )

    # Disjoint by construction (single area covers full grid). Ensure the model
    # uses the disjoint fast-path and that Area.mutate_cells updates area counts.
    model._no_overlap = True

    area = next(a for a in model.areas if a.unique_id != -1)
    voted = np.arange(model.num_colors, dtype=np.int64)

    def _stub_area_step():
        # Minimal "election outcome" needed by mutate_cells.
        area._voter_turnout = 100
        area._voted_ordering = voted

    area.step = _stub_area_step  # type: ignore[method-assign]

    # After init, global_color_dst must match realized grid.
    np.testing.assert_allclose(model.global_color_dst, _grid_distribution_from_cells(model), rtol=0, atol=0)

    # Step 1: no mutation yet (mutation happens at start of step 2).
    model.scheduler.step()
    np.testing.assert_allclose(model.global_color_dst, _grid_distribution_from_cells(model), rtol=0, atol=0)

    # Step 2: mutation from step 1 is applied before the next election.
    before = np.array([c.color for c in model.color_cells], dtype=np.int64)
    model.scheduler.step()
    after = np.array([c.color for c in model.color_cells], dtype=np.int64)

    # With mu=1 and turnout>0, we expect a full recolor (almost surely different from before).
    assert after.shape == before.shape
    assert not np.array_equal(after, before)

    # Exactness invariant: global_color_dst matches realized grid (not preset / stale).
    np.testing.assert_allclose(model.global_color_dst, _grid_distribution_from_cells(model), rtol=0, atol=0)

