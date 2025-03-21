import unittest
import numpy as np
from unittest.mock import MagicMock
from tests.factory import create_default_model

class TestUpdateColorDistribution(unittest.TestCase):
    def setUp(self):
        self.model = create_default_model(
            num_areas=1,
            num_colors=3
        )
        self.model.initialize_area = MagicMock()

    def test_color_distribution(self):
        area = self.model.areas[0]
        old_dist = np.copy(area._color_distribution)
        # Manually change some cell colors
        for cell in area.cells[:3]:
            cell.color = 1
        area._update_color_distribution()
        new_dist = area._color_distribution
        self.assertFalse(np.array_equal(old_dist, new_dist))
        self.assertAlmostEqual(np.sum(new_dist), 1.0, places=5)
