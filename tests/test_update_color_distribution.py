import unittest
import numpy as np
from unittest.mock import MagicMock
from tests.factory import create_test_model

class TestUpdateColorDistribution(unittest.TestCase):
    def setUp(self):
        self.model, _ = create_test_model(
            num_areas=1,
            num_colors=3
        )
        self.model.initialize_area = MagicMock()

    def test_color_distribution(self):
        area = self.model.areas[0]
        old_dist = np.copy(area._color_distribution)
        # Force all cells to color 1
        for cell in area.cells:
            cell.color = 1
        area._update_color_distribution()
        new_dist = area._color_distribution
        # Assert that distribution has changed
        self.assertFalse(np.array_equal(old_dist, new_dist))
        # Assert it's a proper probability distribution
        self.assertAlmostEqual(np.sum(new_dist), 1.0, places=5)
        # Stronger: check all mass on color 1
        self.assertTrue(np.isclose(new_dist[1], 1.0))
