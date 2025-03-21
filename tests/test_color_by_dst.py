import unittest
import numpy as np
from democracy_sim.participation_model import ParticipationModel

class TestColorByDst(unittest.TestCase):

    def test_valid_output(self):
        """Test that the function always returns a valid index."""
        color_distribution = np.array([0.2, 0.3, 0.5])
        for _ in range(1000):
            result = ParticipationModel.color_by_dst(color_distribution)
            self.assertIn(result, range(len(color_distribution)),
                          "Output index is out of range")

    def test_sum_to_one(self):
        """Test that it correctly handles distributions summing to one."""
        color_distribution = np.array([0.1, 0.2, 0.1, 0.4, 0.2])
        for _ in range(50):
            result = ParticipationModel.color_by_dst(color_distribution)
            self.assertIn(result, range(len(color_distribution)))

    def test_single_color(self):
        """Test that a single-color distribution always returns index 0."""
        color_distribution = np.array([1.0])
        for _ in range(10):
            self.assertEqual(
                ParticipationModel.color_by_dst(color_distribution), 0)

    def test_edge_cases(self):
        """Test edge cases like a uniform distribution."""
        color_distribution = np.array([0.5, 0.5])
        results = [ParticipationModel.color_by_dst(
            color_distribution) for _ in range(1000)]
        unique, counts = np.unique(results, return_counts=True)
        self.assertEqual(set(unique), {0, 1},
                         "Function should only return 0 or 1")
        self.assertGreater(int(counts[0]), 400, "Dst not uniform")
        self.assertGreater(int(counts[1]), 400, "Dst not uniform")

    def test_invalid_distribution(self):
        """Test that an invalid distribution raises an error."""
        with self.assertRaises(ValueError):  # Negative probability
            ParticipationModel.color_by_dst(np.array([-0.1, 0.3, 0.8]))

        with self.assertRaises(ValueError):  # Doesn't sum to 1
            ParticipationModel.color_by_dst(np.array([0.2, 0.3]))

        with self.assertRaises(ValueError):  # All zeros
            ParticipationModel.color_by_dst(np.array([0.0, 0.0, 0.0]))

    def test_probability_distribution(self):
        """Test if the function follows the given probability distribution."""
        color_distribution = np.array([0.2, 0.3, 0.5])
        num_samples = 10000
        results = [ParticipationModel.color_by_dst(
            color_distribution) for _ in range(num_samples)]
        
        counts = np.bincount(results,
                             minlength=len(color_distribution)) / num_samples
        err_message = "Generated samples do not match expected distribution"
        np.testing.assert_almost_equal(counts, color_distribution, decimal=1, 
                                       err_msg=err_message)

if __name__ == '__main__':
    unittest.main()
