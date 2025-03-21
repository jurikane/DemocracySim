import unittest
import numpy as np
from itertools import permutations
from tests.factory import create_default_model
from unittest.mock import MagicMock


class TestParticipationModel(unittest.TestCase):

    def setUp(self):
        """Create a fresh model instance before each test and mock `initialize_area`."""
        self.model = create_default_model(
            height=10, width=10, num_agents=100, num_colors=4,
            num_personalities=10, area_size_variance=0.2,
            num_areas=4, av_area_height=5, av_area_width=5,
            heterogeneity=0.5,
        )
        self.model.initialize_area = MagicMock()


    def test_create_personalities_shape(self):
        """Test that the generated personalities array has the correct shape."""
        for n_personalities in range(2, 15):
            personalities = self.model.create_personalities(n_personalities)
            self.assertEqual(personalities.shape,
                             (n_personalities, self.model.num_colors))

    def test_create_personalities_uniqueness(self):
        """Test that the generated personalities are unique."""
        n_personalities = 12
        personalities = self.model.create_personalities(n_personalities)
        unique_personalities = set(map(tuple, personalities))
        self.assertEqual(len(unique_personalities), n_personalities)

    def test_create_personalities_max_limit(self):
        """Test that the method raises an error when
        n exceeds the total number of permutations."""
        assert self.model.num_colors == 4  # 4! = 24 unique permutations
        n_personalities = 25
        with self.assertRaises(ValueError):
            self.model.create_personalities(n_personalities)

    def test_create_personalities_minimum_input(self):
        """Test that the method can handle generating a single personality."""
        personalities = self.model.create_personalities(1)
        self.assertEqual(personalities.shape, (1, self.model.num_colors))

    def test_create_personalities_full_permutation(self):
        """Test that generating the full set of permutations does return all."""
        num_colors = self.model.num_colors
        n_personalities = np.math.factorial(num_colors)
        personalities = self.model.create_personalities(n_personalities)
        expected_permutations = set(permutations(range(num_colors)))
        self.assertEqual(set(map(tuple, personalities)), expected_permutations)


if __name__ == '__main__':
    unittest.main()