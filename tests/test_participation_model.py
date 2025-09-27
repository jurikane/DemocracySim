import unittest
from src.models.participation_model import (ParticipationModel, Area,
                                            distance_functions,
                                            social_welfare_functions)
from src.config.loader import load_config
import mesa

config = load_config()
model_cfg = config.model.model_dump()
vis_cfg = config.visualization.model_dump()


class TestParticipationModel(unittest.TestCase):

    def setUp(self):
        self.model = ParticipationModel(**model_cfg)

    # def test_empty_model(self):
    #     # TODO: Test empty model
    #     model = ParticipationModel(10, 10, 0, 1, 0, 1, 0, 1, 1, 0.1, 1, 0, False, 1, 1, 1, 1, 1, False)
    #     self.assertEqual(model.num_agents, 0)

    def test_initialization(self):
        areas_count = len([area for area in self.model.areas
                           if isinstance(area, Area)])
        self.assertEqual(areas_count, self.model.num_areas)
        self.assertIsInstance(self.model.datacollector, mesa.DataCollector)
        # TODO ... more tests

    def test_model_options(self):
        self.assertEqual(self.model.num_agents, model_cfg["num_agents"])
        self.assertEqual(self.model.num_colors, model_cfg["num_colors"])
        self.assertEqual(self.model.num_areas, model_cfg["num_areas"])
        self.assertEqual(self.model.area_size_variance,
                         model_cfg["area_size_variance"])
        v_rule = social_welfare_functions[model_cfg["rule_idx"]]
        dist_func = distance_functions[model_cfg["distance_idx"]]
        self.assertEqual(self.model.common_assets, model_cfg["common_assets"])
        self.assertEqual(self.model.voting_rule, v_rule)
        self.assertEqual(self.model.distance_func, dist_func)
        self.assertEqual(self.model.election_costs, model_cfg["election_costs"])

    def test_create_color_distribution(self):
        eq_dst = self.model.create_color_distribution(heterogeneity=0)
        self.assertEqual([1/model_cfg["num_colors"] for _ in eq_dst], eq_dst)
        print(f"Color distribution with heterogeneity=0: {eq_dst}")
        het_dst = self.model.create_color_distribution(heterogeneity=1)
        print(f"Color distribution with heterogeneity=1: {het_dst}")
        mid_dst = self.model.create_color_distribution(heterogeneity=0.5)
        print(f"Color distribution with heterogeneity=0.5: {mid_dst}")
        assert het_dst != eq_dst
        assert mid_dst != eq_dst
        assert het_dst != mid_dst

    def test_distribution_of_personalities(self):
        p_dist = self.model.personality_distribution
        self.assertAlmostEqual(sum(p_dist), 1.0)
        self.assertEqual(len(p_dist), model_cfg["num_personalities"])
        voting_agents = self.model.voting_agents
        nr_agents = self.model.num_agents
        personalities = list(self.model.personalities)
        p_counts = {str(i): 0 for i in personalities}
        # Count the occurrence of each personality
        for agent in voting_agents:
            p_counts[str(agent.personality)] += 1
        # Normalize the counts to get the real personality distribution
        real_dist = [p_counts[str(p)] / nr_agents for p in personalities]
        # Simple tests
        self.assertEqual(len(real_dist), len(p_dist))
        self.assertAlmostEqual(float(sum(real_dist)), 1.0)
        # Compare each value
        my_delta = 0.4 / model_cfg["num_personalities"]  # The more personalities, the smaller the delta
        for p_dist_val, real_p_dist_val in zip(p_dist, real_dist):
            self.assertAlmostEqual(p_dist_val, real_p_dist_val, delta=my_delta)


    def test_initialize_areas(self):
        # TODO (very non-trivial) - has been tested manually so far.
        pass

    def test_step(self):
        pass
    # TODO add test_step
    # def test_step(self):
    #     initial_data = self.model.datacollector.get_model_vars_dataframe().copy()
    #     self.model.step()
    #     new_data = self.model.datacollector.get_model_vars_dataframe()
    #     self.assertNotEqual(initial_data, new_data)
