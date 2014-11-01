'''
Smoother unit tests

'''
import unittest

from ace import smoother

class TestSmoother(unittest.TestCase):
    def setUp(self):
        self.smoother = smoother.BasicFixedSpanSmoother()

        self.xData = [1.0, 2.0, 3.0]
        self.yData = [4.0, 5.0, 6.0]
        self.smoother.specify_data_set(self.xData, self.yData)
        self.smoother.window_size = 3
        self.smoother._update_values_in_window()
        self.smoother._update_mean_in_window()
        self.smoother._update_variance_in_window()


    def test_mean(self):

        self.assertAlmostEqual(self.smoother._mean_x_in_window,
                               sum(self.xData) / len(self.xData))

        self.assertAlmostEqual(self.smoother._mean_y_in_window,
                               sum(self.yData) / len(self.yData))

    def test_mean_on_addition_of_observation(self):
        """
        Make sure things work when we add an observation
        """
        self.smoother._add_observation_to_means(7, 8)

        self.assertAlmostEqual(self.smoother._mean_x_in_window,
                               (sum(self.xData) + 7.0) /
                               (self.smoother.window_size + 1.0))

        self.assertAlmostEqual(self.smoother._mean_y_in_window,
                               (sum(self.yData) + 8.0) /
                               (self.smoother.window_size + 1.0))

    def test_mean_on_removal_of_observation(self):
        """
        Make sure things work when we remove an observation
        """
        self.smoother._remove_observation_from_means(3, 6)

        self.assertAlmostEqual(self.smoother._mean_x_in_window,
                               sum(self.xData[:2]) /
                               (self.smoother.window_size - 1.0))

        self.assertAlmostEqual(self.smoother._mean_y_in_window,
                               (sum(self.yData[:2])) /
                               (self.smoother.window_size - 1.0))

    def test_variance_on_removal_of_observation(self):
        """
        Make sure variance and covariance work when we remove an observation quickly
        """
        self.smoother._remove_observation(3, 6)

        cov_from_update = self.smoother._covariance_in_window
        var_from_update = self.smoother._variance_in_window

        self.smoother._update_values_in_window()
        self.smoother._update_mean_in_window()
        self.smoother._update_variance_in_window()

        self.assertAlmostEqual(cov_from_update, self.smoother._covariance_in_window)
        self.assertAlmostEqual(var_from_update, self.smoother._variance_in_window)

    def test_variance_on_addition_of_observation(self):
        """
        Make sure variance and covariance work when we remove an observation quickly
        """
        self.smoother._add_observation(7, 8)
        self.smoother.x.append(7)
        self.smoother.y.append(8)
        cov_from_update = self.smoother._covariance_in_window
        var_from_update = self.smoother._variance_in_window

        self.smoother._update_values_in_window()
        self.smoother._update_mean_in_window()
        self.smoother._update_variance_in_window()

        self.assertAlmostEqual(cov_from_update, self.smoother._covariance_in_window)
        self.assertAlmostEqual(var_from_update, self.smoother._variance_in_window)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
