import unittest
from democracy_sim.participation_model import ParticipationModel, Area
from democracy_sim.model_setup import (grid_rows as height, grid_cols as width,
                                       num_agents, num_colors, num_areas,
                                       draw_borders, rule_idx, voting_rules,
                                       distance_idx, distance_functions,
                                       color_heterogeneity as heterogeneity,
                                       color_patches_steps, av_area_height,
                                       av_area_width, area_size_variance,
                                       patch_power, election_costs)
import mesa


class TestParticipationModel(unittest.TestCase):

    def setUp(self):
        voting_rule = voting_rules[rule_idx]
        distance_func = distance_functions[distance_idx]
        self.model = ParticipationModel(height=height, width=width,
                                        num_agents=num_agents,
                                        num_colors=num_colors,
                                        num_areas=num_areas,
                                        draw_borders=draw_borders,
                                        election_costs=election_costs,
                                        voting_rule=voting_rule,
                                        distance_func=distance_func,
                                        heterogeneity=heterogeneity,
                                        color_patches_steps=color_patches_steps,
                                        av_area_height=av_area_height,
                                        av_area_width=av_area_width,
                                        area_size_variance=area_size_variance,
                                        patch_power=patch_power)

    def test_initialization(self):
        areas_count = len([area for area in self.model.area_scheduler.agents
                           if isinstance(area, Area)])
        self.assertEqual(areas_count, self.model.num_areas)
        self.assertIsInstance(self.model.datacollector, mesa.DataCollector)

    def test_options(self):
        self.assertEqual(self.model.av_area_width, av_area_width)
        self.assertEqual(self.model.area_size_variance, area_size_variance)

    # def test_data_collection(self):
    #     self.model.datacollector.collect(self.model)
    #     data = self.model.datacollector.get_model_vars_dataframe()
    #     self.assertIn("Collective assets", data.columns)
    #     self.assertIn("Number of agents", data.columns)
    #     self.assertIn("Area Color Distributions", data.columns)
    #
    # def test_color_distribution(self):
    #     distribution = self.model.create_color_distribution(heterogeneity=0.5)
    #     self.assertEqual(len(distribution), self.model.num_colors)
    #     self.assertAlmostEqual(sum(distribution), 1.0, places=5)
    #
    # def test_color_patches(self):
    #     from democracy_sim.participation_agent import ColorCell
    #     cell = ColorCell(1, model=self.model, pos=(0, 0), initial_color=0)
    #     color = self.model.color_patches(cell, patch_power=1.0)
    #     self.assertIn(color, range(self.model.num_colors))
    #
    def test_step(self):
        initial_data = self.model.datacollector.get_model_vars_dataframe().copy()
        self.model.step()
        new_data = self.model.datacollector.get_model_vars_dataframe()
        self.assertNotEqual(initial_data, new_data)
