import unittest
import random
import numpy as np
from democracy_sim.participation_model import Area
from democracy_sim.participation_agent import VoteAgent
from .test_participation_model import TestParticipationModel, num_agents
from democracy_sim.social_welfare_functions import majority_rule, approval_voting
from democracy_sim.distance_functions import kendall_tau, spearman


class TestArea(unittest.TestCase):

    def setUp(self):
        test_model = TestParticipationModel()
        test_model.setUp()
        self.model = test_model.model

    def test_update_color_distribution(self):
        rand_area = random.sample(self.model.areas, 1)[0]
        init_dst = rand_area.color_distribution.copy()
        print(f"Area {rand_area.unique_id}s initial color dist.: {init_dst}")
        # Assign new (randomly chosen) cells to the area
        all_color_cells = self.model.color_cells
        rand_area.cells = random.sample(all_color_cells, len(rand_area.cells))
        # Run/test the update_color_distribution method
        rand_area.update_color_distribution()
        new_dst = rand_area.color_distribution
        print(f"Area {rand_area.unique_id}s new color distribution: {new_dst}")
        # Check if the distribution has changed
        assert not np.array_equal(init_dst, new_dst), \
            "Error: The color distribution did not change"

    def test_filter_cells(self):
        # Get existing area
        existing_area = random.sample(self.model.areas, 1)[0]
        print(f"The areas color-cells: "
              f"{[c.unique_id for c in existing_area.cells]}")
        area_cell_sample = random.sample(existing_area.cells, 4)
        other_cells = random.sample(self.model.color_cells, 4)
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

    def test_conduct_election(self):
        area = random.sample(self.model.areas, 1)[0]
        # Test with majority_rule and spearman
        self.model.voting_rule = majority_rule
        self.model.distance_func = spearman
        area.conduct_election()
        # Test with approval_voting and spearman
        self.model.voting_rule = approval_voting
        area.conduct_election()
        # Test with approval_voting and kendall_tau
        self.model.distance_func = kendall_tau
        area.conduct_election()
        # Test with majority_rule and kendall_tau
        self.model.voting_rule = majority_rule
        area.conduct_election()
        # TODO

    def test_adding_new_area_and_agent_within_it(self):
        # Additional area and agent
        personality = random.choice(self.model.personalities)
        a = VoteAgent(num_agents + 1, self.model, pos=(0, 0),
                      personality=personality, assets=25)
        additional_test_area = Area(self.model.num_areas + 1,
                                    model=self.model, height=5,
                                    width=5, size_variance=0)
        additional_test_area.idx_field = (0, 0)  # Place the area at (0, 0)
        test_area = additional_test_area
        print(f"Test-Area: id={test_area.unique_id}, width={test_area._width},"
              f" height={test_area._height}, idx={test_area.idx_field}")
        assert a in test_area.agents  # Test if agent is present
        print(f"Agent {a.unique_id} is in area {test_area.unique_id}")
        print(f"Areas color-cells: {[c.unique_id for c in test_area.cells]}")

    def test_estimate_real_distribution(self):
        # Get any existing area
        rnd_area = random.sample(self.model.areas, 1)[0]
        a = random.sample(rnd_area.agents, 1)[0]
        # Test the estimate_real_distribution method
        a.update_known_cells(area=rnd_area)
        k = len(a.known_cells)
        print(f"Sample size: {k}")
        a_colors = [c.color for c in a.known_cells]  # To test against
        print(f"Cells that agent {a.unique_id} knows of:\n"
              f"{[c.unique_id for c in a.known_cells]} with colors: {a_colors}")
        filtered = rnd_area.filter_cells(a.known_cells)
        select_wrong = [c not in filtered for c in a.known_cells]
        wrong = [c.unique_id for i, c in enumerate(a.known_cells)
                 if select_wrong[i]]
        assert not any(wrong), f"Error: Cells {wrong} are not part of the area!"
        est_distribution, conf = a.estimate_real_distribution(rnd_area)
        assert 0.0 < conf < 1.0, "Error: Confidence out of range [0, 1]!"
        print(f"{a.unique_id}s' estimated color dist is: {est_distribution}",
              f"with confidence: {conf}")
        self.assertAlmostEqual(sum(est_distribution), 1.0, places=7)
        len_colors = self.model.num_colors
        self.assertEqual(len(est_distribution), len_colors)
        counts = [a_colors.count(color) for color in range(len_colors)]
        print(f"Color counts: {counts}")
        s = sum(counts)
        expected_distribution = [i / s for i in counts]
        print(f"Expected distribution: {expected_distribution}")
        self.assertEqual(list(est_distribution), expected_distribution)
