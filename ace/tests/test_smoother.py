"""Smoother unit tests"""

import unittest

from ace import smoother

# pylint: disable=protected-access, missing-docstring

class TestSmoother(unittest.TestCase):
    def setUp(self):
        self.smoother = smoother.BasicFixedSpanSmoother()
        self.x_data = [1.0, 2.0, 3.0, 4.0]
        self.y_data = [4.0, 5.0, 6.0, 7.0]
        self.smoother.specify_data_set(self.x_data, self.y_data)
        self.smoother.window_size = 3
        self.smoother._update_values_in_window()
        self.smoother._update_mean_in_window()
        self.smoother._update_variance_in_window()


    def test_mean(self):
        size = self.smoother.window_size
        self.assertAlmostEqual(self.smoother._mean_x_in_window,
                               sum(self.x_data[:size]) / len(self.x_data[:size]))

        self.assertAlmostEqual(self.smoother._mean_y_in_window,
                               sum(self.y_data[:size]) / len(self.y_data[:size]))

    def test_mean_on_addition_of_observation(self):
        """Make sure things work when we add an observation."""
        self.smoother._add_observation_to_means(7, 8)
        size = self.smoother.window_size
        self.assertAlmostEqual(self.smoother._mean_x_in_window,
                               (sum(self.x_data[:size]) + 7.0) /
                               (self.smoother.window_size + 1.0))

        self.assertAlmostEqual(self.smoother._mean_y_in_window,
                               (sum(self.y_data[:size]) + 8.0) /
                               (self.smoother.window_size + 1.0))

    def test_mean_on_removal_of_observation(self):
        """Make sure things work when we remove an observation."""
        self.smoother._remove_observation_from_means(3, 6)

        self.assertAlmostEqual(self.smoother._mean_x_in_window,
                               sum(self.x_data[:2]) /
                               (self.smoother.window_size - 1.0))

        self.assertAlmostEqual(self.smoother._mean_y_in_window,
                               (sum(self.y_data[:2])) /
                               (self.smoother.window_size - 1.0))

    def test_variance_on_removal_of_observation(self):
        """Make sure variance and covariance work when we remove an observation quickly."""
        self.smoother._remove_observation(3, 6)

        cov_from_update = self.smoother._covariance_in_window
        var_from_update = self.smoother._variance_in_window

        self.smoother._update_values_in_window()
        self.smoother._update_mean_in_window()
        self.smoother._update_variance_in_window()

        self.assertAlmostEqual(cov_from_update, self.smoother._covariance_in_window)
        self.assertAlmostEqual(var_from_update, self.smoother._variance_in_window)

    def test_variance_on_addition_of_observation(self):
        """Make sure variance and covariance work when we remove an observation quickly."""
        self.smoother._add_observation(4, 7)
        cov_from_update = self.smoother._covariance_in_window
        var_from_update = self.smoother._variance_in_window

        self.smoother._update_values_in_window()
        self.smoother._update_mean_in_window()
        self.smoother._update_variance_in_window()

        self.assertAlmostEqual(cov_from_update, self.smoother._covariance_in_window)
        self.assertAlmostEqual(var_from_update, self.smoother._variance_in_window)

    def test_advance_window(self):
        self.assertIn(1.0, self.smoother._x_in_window)
        lowerbound = self.smoother._window_bound_lower
        self.smoother._advance_window()
        self.assertEqual(lowerbound + 1, self.smoother._window_bound_lower)
        self.assertNotIn(1.0, self.smoother._x_in_window)
        self.assertIn(4.0, self.smoother._x_in_window)

    def test_compute_smooth_during_construction(self):
        # test data is linear, so we just make sure we're on the line
        val = self.smoother._compute_smooth_during_construction(2.5)
        self.assertAlmostEqual(val, 5.5)

    def test_compute_cross_validated_residual_here(self):
        # weak test. Hard to do analytically without just repeating the method
        residual = self.smoother._compute_cross_validated_residual_here(2.5, 5.6, 5.5)
        self.assertNotEqual(residual, 0.0)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
