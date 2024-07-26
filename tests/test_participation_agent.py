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

    def test_update_color_distribution(self):
        rand_area = random.sample(self.model.area_scheduler.agents, 1)[0]
        init_dst = rand_area.color_distribution.copy()
        print(f"Area {rand_area.unique_id}s initial color dist.: {init_dst}")
        # Assign new (randomly chosen) cells to the area
        all_color_cells = self.model.color_cell_scheduler.agents
        rand_area.cells = random.sample(all_color_cells, len(rand_area.cells))
        # Run/test the update_color_distribution method
        rand_area.update_color_distribution()
        new_dst = rand_area.color_distribution
        print(f"Area {rand_area.unique_id}s new color distribution: {new_dst}")
        # Check if the distribution has changed
        assert init_dst != new_dst

    def test_filter_cells(self):
        # Get existing area
        existing_area = random.sample(self.model.area_scheduler.agents, 1)[0]
        print(f"The areas color-cells: "
              f"{[c.unique_id for c in existing_area.cells]}")
        area_cell_sample = random.sample(existing_area.cells, 4)
        other_cells = random.sample(self.model.color_cell_scheduler.agents, 4)
        raw_cell_list = area_cell_sample + other_cells
        print(f"Cells to be filtered: {[c.unique_id for c in raw_cell_list]}")
        filtered_cells = existing_area.filter_cells(raw_cell_list)
        print(f"Filtered cells:       {[c.unique_id for c in filtered_cells]}")
        # Check if the cells are filtered correctly
        add_cells = existing_area.filter_cells(other_cells)
        if len(add_cells) > 0:
            print(f"Additional cells: {[c.unique_id for c in add_cells]}")
            area_cell_sample += add_cells
        self.assertEqual(area_cell_sample, filtered_cells)

    def test_estimate_real_distribution(self):
        # Get any existing area
        existing_area = random.sample(self.model.area_scheduler.agents, 1)[0]
        # Create test area
        test_area = Area(self.model.num_areas + 1, model=self.model, height=5,
                         width=5, size_variance=0)
        print(f"Test-Area: id={test_area.unique_id}, width={test_area._width},"
              f" height={test_area._height}, idx={test_area.idx_field}")
        a = self.agent
        test_area.idx_field = (0, 0)
        assert a in test_area.agents  # Test if agent is present
        print(f"Agent {a.unique_id} is in area {test_area.unique_id}")
        print(f"Areas color-cells: {[c.unique_id for c in test_area.cells]}")
        # Test the estimate_real_distribution method
        k = random.sample(range(2, len(test_area.cells)), 1)[0]
        print(f"Sample size: {k}")
        sample_1 = random.sample(test_area.cells, k)
        sample_2 = random.sample(existing_area.cells, 3)
        a.known_cells = sample_1 + sample_2
        a_colors = [c.color for c in a.known_cells]  # To test against
        print(f"Cells that agent {a.unique_id} knows of:\n"
              f"{[c.unique_id for c in a.known_cells]} with colors: {a_colors}")
        print(f"Cells not part of the area: {[c.unique_id for c in sample_2]}")
        rel_cells = test_area.filter_cells(a.known_cells)
        rel_color_vec = [c.color for c in rel_cells]
        print("The relevant cells should be:\n",
              [c.unique_id for c in rel_cells], "with colors", rel_color_vec)
        est_distribution = a.estimate_real_distribution(test_area)
        print(f"Agent {a.unique_id}s' estimated color distribution is: "
              f"{est_distribution}")
        len_colors = self.model.num_colors
        self.assertEqual(len(est_distribution), len_colors)
        counts = [rel_color_vec.count(color) for color in range(len_colors)]
        print(f"Color counts: {counts}")
        s = sum(counts)
        expected_distribution = [i / s for i in counts]
        print(f"Expected distribution: {expected_distribution}")
        self.assertEqual(list(est_distribution), expected_distribution)
