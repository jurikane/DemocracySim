import mesa
import unittest
import numpy as np
from src.models.participation_model import (
    ParticipationModel, Area,
    distance_functions,
    social_welfare_functions
)
from tests.factory import create_test_model


class TestParticipationModelUnit(unittest.TestCase):
    def setUp(self):
        self.model, self.model_cfg = create_test_model()

    ###################################
    # Basic unit tests for Area class #
    ###################################

    def test_initialization_creates_expected_components(self):
        model_cfg = self.model_cfg
        self.assertEqual(self.model.grid.width, model_cfg["width"])
        self.assertEqual(self.model.grid.height, model_cfg["height"])
        self.assertEqual(len(self.model.voting_agents), model_cfg["num_agents"])
        self.assertEqual(len(self.model.color_cells),
                         model_cfg["width"] * model_cfg["height"])
        self.assertEqual(len(self.model.areas), model_cfg["num_areas"])
        self.assertIsNotNone(self.model.global_area)

    def test_personality_group_distribution_sums_to_one(self):
        dist = self.model.personality_group_distribution
        np.testing.assert_almost_equal(dist.sum(), 1.0)
        self.assertEqual(len(dist), len(self.model.personality_groups))

    def test_preset_color_distribution_valid(self):
        dst = self.model.preset_color_dst
        np.testing.assert_almost_equal(sum(dst), 1.0)
        self.assertTrue(all(p >= 0 for p in dst))

    # --- Static & helper methods ---

    def test_color_by_dst_respects_distribution(self):
        probs = np.array([0.1, 0.3, 0.6])
        self.model.np_random = np.random.default_rng(0)
        counts = [0, 0, 0]
        for _ in range(1000):
            c = self.model.color_by_dst_rng(probs)
            counts[c] += 1
        self.assertGreater(counts[2], counts[1])
        self.assertGreater(counts[1], counts[0])

    def test_create_all_options_without_ties(self):
        opts = ParticipationModel.create_all_options(3)
        self.assertEqual(opts.shape[1], 3)
        for row in opts:
            self.assertEqual(sorted(row), [0, 1, 2])

    def test_create_all_options_with_ties(self):
        opts = ParticipationModel.create_all_options(2, include_ties=True)
        self.assertIsInstance(opts, np.ndarray)
        self.assertEqual(opts.shape[1], 2)

    def test_pers_dist_sums_to_one(self):
        dist = ParticipationModel.pers_dist(5, rng=self.model.np_random)
        np.testing.assert_almost_equal(dist.sum(), 1.0)

    # --- Functional behavior ---

    def test_step_updates_model(self):
        before = self.model.av_area_color_dst.copy()
        self.model.step()
        after = self.model.av_area_color_dst
        self.assertEqual(len(before), len(after))
        np.testing.assert_almost_equal(after.sum(), 1.0, decimal=6)

    def test_update_av_area_color_dst(self):
        self.model.update_av_area_color_dst()
        dst = self.model.av_area_color_dst
        np.testing.assert_almost_equal(dst.sum(), 1.0)

    def test_init_color_probs(self):
        probs = self.model.init_color_probs(1.0)
        self.assertEqual(probs.shape, (self.model.num_colors,))
        np.testing.assert_almost_equal(probs.sum(), 1.0)

    def test_initialize_area_adds_area(self):
        old_num = sum(a is not None for a in self.model.areas)
        self.model.initialize_area(0, 0, 0)
        new_num = sum(a is not None for a in self.model.areas)
        self.assertGreaterEqual(new_num, old_num)

    ################################################################
    # Integration tests for Area within ParticipationModel context #
    ################################################################

    def test_initialization(self):
        areas_count = len([
            area for area in self.model.areas if isinstance(area, Area)])
        self.assertEqual(areas_count, self.model.num_areas)
        self.assertIsInstance(self.model.datacollector, mesa.DataCollector)

    def test_model_options(self):
        self.assertEqual(self.model.num_agents, self.model_cfg["num_agents"])
        self.assertEqual(self.model.num_colors, self.model_cfg["num_colors"])
        self.assertEqual(self.model.num_areas, self.model_cfg["num_areas"])
        self.assertEqual(self.model.area_size_variance,
                         self.model_cfg["area_size_variance"])

        v_rule = social_welfare_functions[self.model_cfg["rule_idx"]]
        dist_func = distance_functions[self.model_cfg["distance_idx"]]

        self.assertEqual(self.model.common_assets,
                         self.model_cfg["common_assets"])
        self.assertEqual(self.model.voting_rule, v_rule)
        self.assertEqual(self.model.distance_func, dist_func)
        self.assertEqual(self.model.election_cost_rate,
                         self.model_cfg["election_cost_rate"])

    def test_create_color_distribution(self):
        eq_dst = self.model.create_color_distribution(heterogeneity=0)
        np.testing.assert_allclose(
            eq_dst, [1 / self.model_cfg["num_colors"]] * len(eq_dst))

        het_dst = self.model.create_color_distribution(heterogeneity=1)
        mid_dst = self.model.create_color_distribution(heterogeneity=0.5)

        self.assertFalse(np.allclose(het_dst, eq_dst))
        self.assertFalse(np.allclose(mid_dst, eq_dst))
        self.assertFalse(np.allclose(het_dst, mid_dst))

    def test_distribution_of_personality_groups(self):
        p_dist = self.model.personality_group_distribution
        self.assertAlmostEqual(float(sum(p_dist)), 1.0)
        self.assertEqual(len(p_dist), self.model_cfg["num_personality_groups"])

        voting_agents = self.model.voting_agents
        nr_agents = self.model.num_agents
        personality_groups = list(self.model.personality_groups)
        p_counts = {str(i): 0 for i in personality_groups}

        for agent in voting_agents:
            p_counts[str(agent.personality_group)] += 1

        real_dist = [p_counts[str(p)] / nr_agents for p in personality_groups]

        self.assertEqual(len(real_dist), len(p_dist))
        self.assertAlmostEqual(float(sum(real_dist)), 1.0)

        my_delta = 0.4 / self.model_cfg["num_personality_groups"]
        for p_dist_val, real_p_dist_val in zip(p_dist, real_dist):
            self.assertAlmostEqual(p_dist_val, real_p_dist_val, delta=my_delta)

    def test_initialize_areas(self):
        # TODO (very non-trivial) - has been tested manually so far.
        pass

    def test_step(self):
        # TODO: Add full step integration test
        pass
