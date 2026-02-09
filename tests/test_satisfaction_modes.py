from __future__ import annotations

import numpy as np

from src.utils.distance_functions import distribution_distance_l1
from tests.factory import create_test_model


def test_satisfaction_modes_and_combination() -> None:
    model, _ = create_test_model(num_agents=50, num_colors=2, num_areas=2, num_personality_groups=2)
    area0 = model.areas[0]
    area1 = model.areas[1]
    agent = area0.agents[0]

    # Set known, simple distributions
    area0._color_distribution = np.asarray([0.25, 0.75], dtype=np.float64)
    area1._color_distribution = np.asarray([0.75, 0.25], dtype=np.float64)
    agent.personal_opt_dist = np.asarray([1.0, 0.0], dtype=np.float64)

    # knowledge: force two known cells of color 0
    area0.cells[0].color = 0
    area0.cells[1].color = 0
    agent.known_cells = [area0.cells[0], area0.cells[1]]

    # Global distribution is mean of areas
    global_dist = np.asarray(getattr(model, "_av_area_color_dst"), dtype=np.float64)

    d_area = distribution_distance_l1(agent.personal_opt_dist, area0.color_distribution)
    d_global = distribution_distance_l1(agent.personal_opt_dist, global_dist)
    d_knowledge = distribution_distance_l1(agent.personal_opt_dist, np.asarray([1.0, 0.0], dtype=np.float64))

    model.satisfaction_mode = "area"
    assert np.isclose(agent.compute_satisfaction_value(area=area0, model=model), d_area)

    model.satisfaction_mode = "global"
    assert np.isclose(agent.compute_satisfaction_value(area=area0, model=model), d_global)

    model.satisfaction_mode = "knowledge"
    assert np.isclose(agent.compute_satisfaction_value(area=area0, model=model), d_knowledge)

    model.satisfaction_mode = "combination"
    expected = (d_area + d_global + d_knowledge) / 3.0
    assert np.isclose(agent.compute_satisfaction_value(area=area0, model=model), expected)

