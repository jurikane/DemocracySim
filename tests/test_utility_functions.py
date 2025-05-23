import unittest
import numpy as np
from democracy_sim.participation_agent import combine_and_normalize

class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions in the democracy_sim package."""

    def test_combine_and_normalize_basic(self):
        """Test basic functionality of combine_and_normalize."""
        # Test with equal arrays
        arr1 = np.array([0.25, 0.25, 0.25, 0.25])
        arr2 = np.array([0.25, 0.25, 0.25, 0.25])
        result = combine_and_normalize(arr1, arr2, 0.5)
        np.testing.assert_array_almost_equal(result, np.array([0.25, 0.25, 0.25, 0.25]))
        
        # Test with factor = 0 (should return arr2)
        result = combine_and_normalize(arr1, arr2, 0.0)
        np.testing.assert_array_almost_equal(result, arr2)
        
        # Test with factor = 1 (should return arr1)
        result = combine_and_normalize(arr1, arr2, 1.0)
        np.testing.assert_array_almost_equal(result, arr1)
        
    def test_combine_and_normalize_different_arrays(self):
        """Test combine_and_normalize with different arrays."""
        arr1 = np.array([0.1, 0.2, 0.3, 0.4])
        arr2 = np.array([0.4, 0.3, 0.2, 0.1])
        
        # Test with factor = 0.5 (should be average)
        result = combine_and_normalize(arr1, arr2, 0.5)
        expected = np.array([0.25, 0.25, 0.25, 0.25])
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test with factor = 0.75 (weighted more toward arr1)
        result = combine_and_normalize(arr1, arr2, 0.75)
        expected = (0.75 * arr1 + 0.25 * arr2) / np.sum(0.75 * arr1 + 0.25 * arr2)
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_combine_and_normalize_normalization(self):
        """Test that the result is properly normalized."""
        arr1 = np.array([1.0, 2.0, 3.0, 4.0])  # Not normalized
        arr2 = np.array([5.0, 6.0, 7.0, 8.0])  # Not normalized
        
        result = combine_and_normalize(arr1, arr2, 0.5)
        self.assertAlmostEqual(np.sum(result), 1.0)
        
    def test_combine_and_normalize_invalid_factor(self):
        """Test that an invalid factor raises a ValueError."""
        arr1 = np.array([0.25, 0.25, 0.25, 0.25])
        arr2 = np.array([0.25, 0.25, 0.25, 0.25])
        
        with self.assertRaises(ValueError):
            combine_and_normalize(arr1, arr2, -0.1)
            
        with self.assertRaises(ValueError):
            combine_and_normalize(arr1, arr2, 1.1)

if __name__ == "__main__":
    unittest.main()