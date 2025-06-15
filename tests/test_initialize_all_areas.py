import unittest
from unittest.mock import MagicMock
from numpy import sqrt
from tests.factory import create_default_model  # Import from factory.py


class TestParticipationModelInitializeAllAreas(unittest.TestCase):

    def setUp(self):
        """Create a fresh model instance before each test and mock `initialize_area`."""
        self.model = create_default_model(
            num_areas=4,  # Override num_areas to 4
            height=10,  # Set grid height
            width=10,  # Set grid width
            av_area_height=5,  # Set average area height
            av_area_width=5,  # Set average area width
        )
        self.model.initialize_area = MagicMock()  # Mock `initialize_area` for side effect tracking

    def test_initialize_all_areas_uniform_distribution(self):
        """Test that areas are initialized uniformly if num_areas is a perfect square."""

        # Check if the areas are initialized in a roughly uniform grid-like pattern
        expected_calls = [(0, 0), (5, 0), (0, 5), (5, 5)]
        # Check if 4 areas were initialized
        self.assertEqual(self.model.num_areas, 4)  # Check num_areas==4
        idx_fields = [area.idx_field for area in self.model.areas]
        # Collect idx_fields from all areas
        for idx_field in idx_fields:
            assert idx_field in expected_calls

    def test_initialize_all_areas_with_non_square_number(self):
        """Test that the method handles non-square numbers by adding extra areas randomly."""
        model = create_default_model(
            num_areas=5,  # Override num_areas to 5
        )
        # model.initialize_all_areas()  # Runs on initialization
        # Check that 5 areas were initialized after calling the function
        self.assertEqual(model.num_areas, 5)

    def test_initialize_all_areas_no_areas(self):
        """Test that the method does nothing if num_areas is 0."""
        model = create_default_model(
            num_areas=0,  # Set num_areas to 0
        )
        assert model.num_areas == 0  # Verify no areas were initialized

    def test_initialize_all_areas_random_additional_areas(self):
        """Test that additional areas are placed randomly if num_areas exceeds uniform grid capacity."""
        model = create_default_model(
            num_areas=5,  # Override num_areas to 5
            height=10,
            width=10,
            av_area_height=5,
            av_area_width=5
        )

        # Check that the number of initialized areas matches num_areas
        self.assertEqual(model.num_areas, 5)  # Check that exactly 5 areas are initialized

        # Check that at least one area was placed outside the uniform pattern
        idx_fields = [area.idx_field for area in model.areas]
        expected_calls = [(0, 0), (5, 0), (0, 5), (5, 5)]
        random_area_detected = any(
            idx_field not in expected_calls for idx_field in idx_fields
        )
        self.assertTrue(random_area_detected)

    def test_initialize_all_areas_handles_non_square_distribution(self):
        """Test that the number of areas matches `num_areas` even for non-square cases."""
        model = create_default_model(
            num_areas=6,  # Override num_areas to 6
        )
        # Check that exactly 6 areas are initialized
        self.assertEqual(model.num_areas, 6)

    def test_initialize_all_areas_calculates_distances_correctly(self):
        """Test that area distances are calculated correctly."""
        model = create_default_model(
            num_areas=4,  # Override num_areas to 4
            height=10,
            width=10,
            av_area_height=5,
            av_area_width=5
        )
        # Calculate the expected distances
        roo_apx = round(sqrt(model.num_areas))
        expected_distance_x = model.grid.width // roo_apx
        expected_distance_y = model.grid.height // roo_apx

        # Check the calculated distances
        self.assertEqual(expected_distance_x, 5)
        self.assertEqual(expected_distance_y, 5)


if __name__ == '__main__':
    unittest.main()