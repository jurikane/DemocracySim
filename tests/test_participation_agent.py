import unittest
from test_participation_model import *
from democracy_sim.participation_model import ParticipationModel, Area
from democracy_sim.participation_agent import VoteAgent, combine_and_normalize
import numpy as np
import random


class TestVotingAgent(unittest.TestCase):

    def setUp(self):
        test_model = TestParticipationModel()
        test_model.setUp()
        self.model = test_model.model
        personality = np.zeros(self.model.num_colors)
        personality[0] = 0.3
        personality[1] = 0.7
        self.agent = VoteAgent(num_agents + 1, (0, 0), self.model,
                               personality=personality, assets=25)

    def test_combine_and_normalize_rank_arrays(self):
        # TODO more test-cases and include estimated results
        print("Test function combine_and_normalize_estimates")
        a = np.array([0.0, 0.2, 0.7, 0.5, 0.1, 0.8, 1.0])
        a_rank = np.argsort(a)
        print(f"Ranking of a: {a_rank}")
        b = np.array([1.0, 0.2, 0.7, 0.5, 0.1, 0.8, 0.0])
        b_rank = np.argsort(b)
        print(f"Ranking of b: {b_rank}")
        factors = [0.0, 0.2, 0.5, 1.0]
        for f in factors:
            result = combine_and_normalize(a, b, f)
            result_rank = np.argsort(result)
            print(f"Ranking of r with factor {f}: {result_rank}")

    def test_estimate_real_distribution(self):
        existing_area = random.sample(self.model.area_scheduler.agents, 1)[0]
        test_area = Area(999, model=self.model, height=5, width=5,
                         size_variance=0)
        test_area.idx_field((0, 0))
        a = self.agent
        assert a in test_area.agents  # Test if agent is present
        print(f"Areas color-cells: {test_area.cells}")
        # Test the estimate_real_distribution method
        known_cells = (random.sample(test_area.cells, 4) +
                       random.sample(existing_area, 4))
        print(f"Cells that agent {a.unique_id} knows of: {a.known_cells}")
        # TODO finish this!

