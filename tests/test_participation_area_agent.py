import unittest
import random
import numpy as np
from src.agents.area import Area
from src.agents.color_cell import ColorCell
from src.agents.vote_agent import VoteAgent
from src.model_setup import build_model_kwargs
from src.models.participation_model import ParticipationModel
from src.config.loader import load_config
from src.utils.social_welfare_functions import majority_rule, approval_voting
from src.utils.distance_functions import kendall_tau, spearman
import mesa.space as space
from mesa import Model


###################################
# Dummy classes for isolated tests #
###################################

class DummyModel(Model):
    """Minimal model for testing Area without full ParticipationModel."""
    def __init__(self, width=5, height=5, num_colors=3):
        super().__init__()
        self.width = width
        self.height = height
        self.num_colors = num_colors
        self.grid = space.SingleGrid(height=height, width=width, torus=True)
        self.personality_groups = [0, 1]
        self.voting_agents = []
        # Stubs for election-related attributes
        self.distance_func = lambda *args, **kwargs: 0
        self.voting_rule = lambda x: [0]
        self.options = [0, 1, 2]
        self.known_cells = 1
        self.color_search_pairs = []
        self.max_reward = 10
        self.election_cost_rate = 1
        self.mu = 0.1
        self.color_probs = [1 / num_colors] * num_colors
        self.np_random = np.random.default_rng()
        self.personal_opt_dist_concentration = 1.0
        self.participation_init_q = 0.0
        self.altruism_init = 0.5
        self.participation_q_max = 50.0
        self.bias_toward_participation = 0.0
        self.altruism_alpha = 0.05
        self.altruism_clip_min = 0.0
        self.altruism_clip_max = 1.0
        self.altruism_learning = False
        self.altruism_static = 0.5


##################################
# Unit tests for the Area class  #
##################################

class TestAreaBasics(unittest.TestCase):

    def setUp(self):
        self.dummy_model = DummyModel()

    def test_initialization_no_variance(self):
        area = Area(unique_id=1, model=self.dummy_model, height=2, width=3, size_variance=0)
        self.assertEqual(area.num_cells, 6)
        self.assertEqual(area.num_agents, 0)
        self.assertTrue((area.color_distribution == np.zeros(self.dummy_model.num_colors)).all())

    def test_invalid_variance_raises(self):
        with self.assertRaises(ValueError):
            Area(unique_id=2, model=self.dummy_model, height=2, width=3, size_variance=1.5)

    def test_add_agent_and_cell(self):
        area = Area(1, self.dummy_model, 2, 2, 0)
        cell = ColorCell(10, self.dummy_model, (0, 0), 1)
        area.add_cell(cell)
        dummy_agent = VoteAgent(1, self.dummy_model, (0, 0),
                                personality_group=list(range(self.dummy_model.num_colors)))
        area.add_agent(dummy_agent)

        self.assertIn(cell, area.cells)
        self.assertIn(dummy_agent, area.agents)
        self.assertEqual(area.num_agents, 1)

    def test_idx_field_assigns_cells(self):
        # Replace the automatically placed cell with our own
        cell = ColorCell(11, self.dummy_model, (0, 0), 2)
        present_cell = self.dummy_model.grid.get_cell_list_contents([(0, 0)])[0]
        self.dummy_model.grid.remove_agent(present_cell)
        self.dummy_model.grid.place_agent(cell, (0, 0))

        area = Area(1, self.dummy_model, 1, 1, 0)
        area.idx_field = (0, 0)

        self.assertEqual(area.idx_field, (0, 0))
        self.assertIn(cell, area.cells)
        self.assertAlmostEqual(area.color_distribution.sum(), 1.0, places=7)

    def test_str_representation(self):
        area = Area(99, self.dummy_model, 2, 3, 0)
        s = str(area)
        self.assertIn("Area(id=99", s)
        self.assertIn("size=2x3", s)
        self.assertIn("num_agents=0", s)
        self.assertIn("num_cells=6", s)


################################################################
# Integration tests for Area within ParticipationModel context #
################################################################

class TestAreaIntegration(unittest.TestCase):

    def setUp(self):
        self.model_cfg = load_config().model
        model_cfg = build_model_kwargs(self.model_cfg)
        self.model = ParticipationModel(**model_cfg)

    def test_update_color_distribution(self):
        rand_area = random.sample(self.model.areas, 1)[0]
        init_dst = rand_area.color_distribution.copy()
        print(f"Area {rand_area.unique_id}s initial color dist.: {init_dst}")
        # Assign new (randomly chosen) cells to the area
        all_color_cells = self.model.color_cells
        rand_area.cells = random.sample(all_color_cells, len(rand_area.cells))
        # Run/test the update_color_distribution method
        rand_area._update_color_distribution()
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
        filtered_cells = existing_area._filter_cells(raw_cell_list)
        print(f"Filtered cells:       {[c.unique_id for c in filtered_cells]}")
        # Check if the cells are filtered correctly
        add_cells = existing_area._filter_cells(other_cells)
        if len(add_cells) > 0:
            print(f"Additional cells: {[c.unique_id for c in add_cells]}")
            area_cell_sample += add_cells
        self.assertEqual(area_cell_sample, filtered_cells)

    def test_conduct_election(self):
        area = random.sample(self.model.areas, 1)[0]
        # Test with majority_rule and spearman
        self.model.voting_rule = majority_rule
        self.model.distance_func = spearman
        for agent in area.agents:
            agent.update_known_cells(area)
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
        personality_group = random.choice(self.model.personality_groups)
        a = VoteAgent(self.model_cfg.num_agents + 1, self.model, pos=(0, 0),
                      personality_group=personality_group, assets=25)
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
        filtered = rnd_area._filter_cells(a.known_cells)
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
