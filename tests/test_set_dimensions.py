import unittest
from tests.factory import create_default_model

class TestSetDimensions(unittest.TestCase):
    def setUp(self):
        self.model = create_default_model(
            num_areas=1,
            height=10,
            width=10,
            av_area_height=5,
            av_area_width=5,
            area_size_variance=0
        )

    def test_dimensions_no_variance(self):
        area = self.model.areas[0]
        self.assertEqual(area._width, 5)
        self.assertEqual(area._height, 5)

    def test_dimensions_out_of_range(self):
        with self.assertRaises(ValueError):
            bad_model = create_default_model(
                num_areas=1,
                av_area_width=5,
                av_area_height=5,
                area_size_variance=2
            )
            _ = bad_model.areas[0]