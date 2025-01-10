import unittest
from democracy_sim.participation_model import (ParticipationModel, Area,
                                               distance_functions,
                                               social_welfare_functions)
from democracy_sim.model_setup import (grid_rows as height, grid_cols as width,
                                       num_agents, num_colors, num_areas,
                                       num_personalities, common_assets, mu,
                                       known_cells,
                                       election_impact_on_mutation as e_impact,
                                       draw_borders, rule_idx, distance_idx,
                                       color_heterogeneity as heterogeneity,
                                       color_patches_steps, av_area_height,
                                       av_area_width, area_size_variance,
                                       patch_power, election_costs, max_reward)
import mesa


class TestParticipationModel(unittest.TestCase):

    def setUp(self):
        self.model = ParticipationModel(height=height, width=width,
                                        num_agents=num_agents,
                                        num_colors=num_colors,
                                        num_personalities=num_personalities,
                                        known_cells=known_cells,
                                        common_assets=common_assets, mu=mu,
                                        election_impact_on_mutation=e_impact,
                                        num_areas=num_areas,
                                        draw_borders=draw_borders,
                                        election_costs=election_costs,
                                        rule_idx=rule_idx,
                                        distance_idx=distance_idx,
                                        heterogeneity=heterogeneity,
                                        color_patches_steps=color_patches_steps,
                                        av_area_height=av_area_height,
                                        av_area_width=av_area_width,
                                        area_size_variance=area_size_variance,
                                        patch_power=patch_power,
                                        max_reward=max_reward,
                                        show_area_stats=False)

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
        self.assertEqual(self.model.num_agents, num_agents)
        self.assertEqual(self.model.num_colors, num_colors)
        self.assertEqual(self.model.num_areas, num_areas)
        self.assertEqual(self.model.area_size_variance, area_size_variance)
        self.assertEqual(self.model.draw_borders, draw_borders)
        v_rule = social_welfare_functions[rule_idx]
        dist_func = distance_functions[distance_idx]
        self.assertEqual(self.model.common_assets, common_assets)
        self.assertEqual(self.model.voting_rule, v_rule)
        self.assertEqual(self.model.distance_func, dist_func)
        self.assertEqual(self.model.election_costs, election_costs)

    def test_create_color_distribution(self):
        eq_dst = self.model.create_color_distribution(heterogeneity=0)
        self.assertEqual([1/num_colors for _ in eq_dst], eq_dst)
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
        self.assertEqual(len(p_dist), num_personalities)
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
        my_delta = 0.4 / num_personalities  # The more personalities, the smaller the delta
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
