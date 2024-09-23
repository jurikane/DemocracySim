import unittest
from democracy_sim.participation_model import ParticipationModel, Area
from democracy_sim.model_setup import (grid_rows as height, grid_cols as width,
                                       num_agents, num_colors, num_areas,
                                       num_personalities,
                                       num_personality_colors as npc,
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
                                        num_personality_colors=npc,
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
                                        max_reward=max_reward)

    def test_initialization(self):
        areas_count = len([area for area in self.model.area_scheduler.agents
                           if isinstance(area, Area)])
        self.assertEqual(areas_count, self.model.num_areas)
        self.assertIsInstance(self.model.datacollector, mesa.DataCollector)
        # TODO ... more tests

    def test_model_options(self):
        self.assertEqual(self.model.num_agents, num_agents)
        self.assertEqual(self.model.num_colors, num_colors)
        self.assertEqual(self.model.num_personalities, num_personalities)
        self.assertEqual(self.model.num_personality_colors, npc)
        self.assertEqual(self.model.num_areas, num_areas)
        self.assertEqual(self.model.area_size_variance, area_size_variance)
        self.assertEqual(self.model.draw_borders, draw_borders)
        self.assertEqual(self.model.heterogeneity, heterogeneity)
        v_rule = social_welfare_functions[rule_idx]
        dist_func = distance_functions[distance_idx]
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
