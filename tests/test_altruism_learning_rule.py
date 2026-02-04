from __future__ import annotations

import numpy as np

from tests.factory import create_test_model


def test_altruism_learning_participant_only_direction_rules() -> None:
    """Contract test for altruism learning (participant-only, directional).

    Rules:
    - participating + positive delta => altruism_factor increases
    - participating + negative delta => altruism_factor decreases
    - not participating => no change

    Update rule:
        a = a + altruism_alpha * delta_assets
        clip to [altruism_clip_min, altruism_clip_max]
    """

    model, _ = create_test_model(
        seed=1,
        num_agents=1,
        num_colors=3,
        num_areas=1,
        altruism_alpha=0.1,
        altruism_init=0.5,
        altruism_clip_min=0.0,
        altruism_clip_max=1.0,
        max_steps=1,
    )

    a = model.voting_agents[0]

    # Participating => directional updates
    a._participating = True

    a.altruism_factor = 0.6
    a.apply_altruism_update(delta_assets=+1.0)
    assert a.altruism_factor > 0.6

    a.altruism_factor = 0.6
    a.apply_altruism_update(delta_assets=-1.0)
    assert a.altruism_factor < 0.6

    a.altruism_factor = 0.4
    a.apply_altruism_update(delta_assets=+1.0)
    assert a.altruism_factor > 0.4

    a.altruism_factor = 0.4
    a.apply_altruism_update(delta_assets=-1.0)
    assert a.altruism_factor < 0.4

    # Not participating => no update
    a.altruism_factor = 0.6
    a._participating = False
    a.apply_altruism_update(delta_assets=+1.0)
    assert np.isclose(a.altruism_factor, 0.6)
