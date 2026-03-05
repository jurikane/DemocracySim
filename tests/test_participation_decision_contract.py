from __future__ import annotations

from tests.factory import create_test_model
from src.agents.strategies import DefaultParticipationStrategy


class _FixedRng:
    def __init__(self, value: float):
        self._value = float(value)

    def random(self) -> float:
        return float(self._value)


def test_participation_decision_uses_probability_and_rng_draw() -> None:
    model, _ = create_test_model(
        seed=40,
        num_agents=1,
        num_colors=3,
        num_areas=1,
        participation_beta=1.0,
    )
    a = model.voting_agents[0]
    assert a is not None
    a.q_participation = 0.0  # sigmoid(0.0) == 0.5

    model.participation_rng = _FixedRng(0.49)
    strat = DefaultParticipationStrategy()
    assert strat.decide_participation(a, area=None) is True

    model.participation_rng = _FixedRng(0.51)
    assert strat.decide_participation(a, area=None) is False


def test_participation_q_controls_decision_for_fixed_draw() -> None:
    m1, _ = create_test_model(seed=41, num_agents=1, num_colors=3, num_areas=1, participation_beta=2.0)
    m2, _ = create_test_model(seed=41, num_agents=1, num_colors=3, num_areas=1, participation_beta=2.0)
    a1 = m1.voting_agents[0]
    a2 = m2.voting_agents[0]
    assert a1 is not None and a2 is not None

    a1.q_participation = -1.0
    a2.q_participation = 1.0
    m1.participation_rng = _FixedRng(0.6)
    m2.participation_rng = _FixedRng(0.6)

    strat = DefaultParticipationStrategy()
    assert strat.decide_participation(a1, area=None) is False
    assert strat.decide_participation(a2, area=None) is True
