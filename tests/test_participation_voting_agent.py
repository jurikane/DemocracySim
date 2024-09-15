from .test_participation_model import *
from democracy_sim.participation_model import Area
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
        self.agent = VoteAgent(num_agents + 1, self.model, pos=(0, 0),
                               personality=personality, assets=25)
        self.additional_test_area = Area(self.model.num_areas + 1,
                                         model=self.model, height=5,
                                         width=5, size_variance=0)
        self.additional_test_area.idx_field = (0, 0)  # Place the area at (0, 0)

    def test_combine_and_normalize_rank_arrays(self):
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

    def test_combine_and_normalize(self):
        a = self.agent
        test_area = self.additional_test_area
        assert a in test_area.agents  # Test if agent is present
        # Give the agent some cells to know of
        k = random.sample(range(2, len(test_area.cells)), 1)[0]
        print(f"Sample size: {k}")
        a.known_cells = random.sample(test_area.cells, k)
        est_dist = a.estimate_real_distribution(test_area)
        own_prefs = a.personality
        # own_prefs = np.array([0.25, 0.5, 0.0, 0.0]) # Should also work..
        print(f"Agent {a.unique_id}s' personality: {own_prefs}"
              f" and estimated color distribution: {est_dist}")
        for a_factor in [0.0, 0.2, 0.5, 1.0]:
            comb = combine_and_normalize(own_prefs, est_dist, a_factor)
            print(f"Assumed opt. distribution with factor {a_factor}: \n{comb}")
            # Validation
            if a_factor == 0.0:
                self.assertEqual(list(comb), list(est_dist))
            elif a_factor == 1.0:
                if sum(own_prefs) != 1.0:
                    own_prefs = own_prefs / sum(own_prefs)
                self.assertEqual(list(comb), list(own_prefs))
            self.assertTrue(np.isclose(sum(comb), 1.0, atol=1e-8))

    def test_compute_assumed_opt_dist(self):
        a = self.agent
        test_area = self.additional_test_area
        # Give the agent some cells to know of
        max_size = len(test_area.cells)
        k = random.sample(range(2, max_size), 1)[0]
        a.known_cells = random.sample(test_area.cells, k=k)
        est_dist = a.estimate_real_distribution(test_area)
        own_prefs = a.personality
        print(f"The agents\npersonality: {own_prefs} \nest_dist   : {est_dist}")
        r = a.compute_assumed_opt_dist(test_area)
        print(f"Assumed optimal distribution: {r}")
        self.assertTrue(np.isclose(sum(r), 1.0, atol=1e-8))



