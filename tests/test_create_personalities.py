import unittest
from math import factorial
from itertools import permutations
from tests.factory import create_test_model
from unittest.mock import MagicMock


class TestParticipationModel(unittest.TestCase):

    def setUp(self):
        """Create a fresh model instance before each test and mock `initialize_area`."""
        self.model, _ = create_test_model(
            height=10, width=10, num_agents=100, num_colors=4,
            num_personality_groups=10, area_size_variance=0.2,
            num_areas=4, av_area_height=5, av_area_width=5,
            heterogeneity=0.5,
        )
        self.model.initialize_area = MagicMock()


    def test_create_personality_groups_shape(self):
        """Test that the generated personality_groups array has the correct shape."""
        for n_personality_groups in range(2, 15):
            personality_groups = self.model.create_personality_groups(n_personality_groups)
            self.assertEqual(personality_groups.shape,
                             (n_personality_groups, self.model.num_colors))

    def test_create_personality_groups_uniqueness(self):
        """Test that the generated personality_groups are unique."""
        n_personality_groups = 12
        personality_groups = self.model.create_personality_groups(n_personality_groups)
        unique_personality_groups = set(map(tuple, personality_groups))
        self.assertEqual(len(unique_personality_groups), n_personality_groups)

    def test_create_personality_groups_max_limit(self):
        """Test that the method raises an error when
        n exceeds the total number of permutations."""
        assert self.model.num_colors == 4  # 4! = 24 unique permutations
        n_personality_groups = 25
        with self.assertRaises(ValueError):
            self.model.create_personality_groups(n_personality_groups)

    def test_create_personality_groups_minimum_input(self):
        """Test that the method can handle generating a single personality_group."""
        personality_groups = self.model.create_personality_groups(1)
        self.assertEqual(personality_groups.shape, (1, self.model.num_colors))

    def test_create_personality_groups_full_permutation(self):
        """Test that generating the full set of permutations does return all."""
        num_colors = self.model.num_colors
        n_personality_groups = factorial(num_colors)
        personality_groups = self.model.create_personality_groups(n_personality_groups)
        expected_permutations = set(permutations(range(num_colors)))
        self.assertEqual(set(map(tuple, personality_groups)), expected_permutations)


if __name__ == '__main__':
    unittest.main()