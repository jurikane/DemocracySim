from __future__ import annotations

import numpy as np
import pytest

from src.utils.distance_functions import distribution_distance_l1
from tests.factory import create_test_model


pytestmark = pytest.mark.phase1


def _make_two_area_model():
    # Deterministic non-overlapping geometry.
    model, _ = create_test_model(
        seed=120,
        num_agents=20,
        num_colors=2,
        num_areas=2,
        num_personality_groups=2,
        height=10,
        width=10,
        av_area_height=10,
        av_area_width=5,
        area_size_variance=0.0,
        mu=0.0,
    )
    assert len(model.areas) == 2
    return model


def test_satisfaction_mode_oracle_matches_expected_targets() -> None:
    """Oracle: each satisfaction_mode uses the intended target distribution."""
    model = _make_two_area_model()
    area0, area1 = model.areas[0], model.areas[1]
    agent = area0.agents[0]

    # Set known, simple distributions.
    area0._color_distribution = np.asarray([0.25, 0.75], dtype=np.float64)
    area1._color_distribution = np.asarray([0.75, 0.25], dtype=np.float64)

    # Personality distribution.
    agent.personal_opt_dist = np.asarray([1.0, 0.0], dtype=np.float64)

    # Knowledge distribution: two known cells of color 0.
    area0.cells[0].color = 0
    area0.cells[1].color = 0
    agent.known_cells = [area0.cells[0], area0.cells[1]]

    # Ensure global distribution reflects area distributions.
    model.update_global_color_distribution()

    d_area = distribution_distance_l1(agent.personality, area0.color_distribution)
    d_global = distribution_distance_l1(agent.personality, model.global_color_dst)
    d_knowledge = distribution_distance_l1(agent.personality, np.asarray([1.0, 0.0], dtype=np.float64))

    model.satisfaction_mode = "area"
    assert float(agent.compute_dissatisfaction_value(area=area0, model=model)) == pytest.approx(d_area, abs=1e-12)

    model.satisfaction_mode = "global"
    assert float(agent.compute_dissatisfaction_value(area=area0, model=model)) == pytest.approx(d_global, abs=1e-12)

    model.satisfaction_mode = "knowledge"
    assert float(agent.compute_dissatisfaction_value(area=area0, model=model)) == pytest.approx(d_knowledge, abs=1e-12)

    model.satisfaction_mode = "combination"
    expected = float((d_area + d_global + d_knowledge) / 3.0)
    assert float(agent.compute_dissatisfaction_value(area=area0, model=model)) == pytest.approx(expected, abs=1e-12)


def test_satisfaction_mode_metamorphic_global_vs_area_ordering() -> None:
    """Metamorphic: if the local area is closer to the agent than the global mean, area sv < global sv."""
    model = _make_two_area_model()
    area0, area1 = model.areas[0], model.areas[1]
    agent = area0.agents[0]
    agent.personal_opt_dist = np.asarray([1.0, 0.0], dtype=np.float64)

    # Make area0 close to personality, area1 far, so global is intermediate.
    area0._color_distribution = np.asarray([0.9, 0.1], dtype=np.float64)
    area1._color_distribution = np.asarray([0.1, 0.9], dtype=np.float64)
    model.update_global_color_distribution()

    model.satisfaction_mode = "area"
    sv_area = float(agent.compute_dissatisfaction_value(area=area0, model=model))
    model.satisfaction_mode = "global"
    sv_global = float(agent.compute_dissatisfaction_value(area=area0, model=model))

    assert sv_area < sv_global


def test_satisfaction_mode_knowledge_requires_nonempty_known_cells() -> None:
    """Fail-loud contract: knowledge mode requires a non-empty known_cells list."""
    model = _make_two_area_model()
    area0 = model.areas[0]
    agent = area0.agents[0]
    agent.known_cells = []
    model.satisfaction_mode = "knowledge"
    with pytest.raises(ValueError, match="no known cells"):
        agent.compute_dissatisfaction_value(area=area0, model=model)

