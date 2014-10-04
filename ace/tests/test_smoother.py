'''
Smoother tests

Note: these aren't real unittests yet. Just making sure we can reproduce 
the plots from the paper. 

'''
import math
import unittest

import numpy
import pylab

from .. import smoother
from .. import supersmoother

class TestSmoother(unittest.TestCase):
    def setUp(self):
        self.smoother = smoother.BasicFixedSpanSmoother()
        self.smoother.specify_data_set(range(5), range(5))
        self.xData = [1.0, 2.0, 3.0]
        self.yData = [4.0, 5.0, 6.0]
        self.smoother._update_mean_in_window(self.xData, self.yData)
        self.smoother._update_variance_in_window(self.xData, self.yData)
        self.smoother.window_size = 3

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
        self.smoother._remove_observation_to_variances(3, 6)
        self.smoother._remove_observation_from_means(3, 6)

        cov_from_update = self.smoother._covariance_in_window
        var_from_update = self.smoother._variance_in_window

        self.smoother._update_mean_in_window(self.xData[:2], self.yData[:2])
        self.smoother._update_variance_in_window(self.xData[:2], self.yData[:2])

        self.assertAlmostEqual(cov_from_update, self.smoother._covariance_in_window)
        self.assertAlmostEqual(var_from_update, self.smoother._variance_in_window)

    def test_variance_on_addition_of_observation(self):
        """
        Make sure variance and covariance work when we remove an observation quickly
        """
        self.smoother._add_observation_to_means(7, 8)
        self.smoother._add_observation_to_variances(7, 8)

        cov_from_update = self.smoother._covariance_in_window
        var_from_update = self.smoother._variance_in_window

        self.smoother._update_mean_in_window(self.xData + [7], self.yData + [8])
        self.smoother._update_variance_in_window(self.xData + [7], self.yData + [8])

        self.assertAlmostEqual(cov_from_update, self.smoother._covariance_in_window)
        self.assertAlmostEqual(var_from_update, self.smoother._variance_in_window)

def build_test_problem(N=200):
    x = numpy.linspace(0, 1, N)
    # add iid standard normal error
    err = numpy.random.standard_normal(N)
    y = [math.sin(2 * math.pi * (1 - xi) ** 2) + xi * ei for xi, ei in zip(x, err)]
    return x, y

class TestProblemSmoothers(unittest.TestCase):

    def setUp(self):
        self.x, self.y = build_test_problem()
        pylab.figure()
        # pylab.plot(self.x, self.y, '.', label='Data')

    @unittest.skip('Plots stuff')
    def test_basic_smoother(self):
        """
        Runs Friedman's test from Figure 2b. 
        """

        for span in smoother.DEFAULT_SPANS:
            smoother = smoother.perform_smooth(self.x, self.y, span)
            pylab.plot(self.x, smoother.smooth_result, label='Span = {0}'.format(span))
        finish_plot()

    @unittest.skip('Plots stuff')
    def test_supersmoother(self):
        my_smoother = smoother.perform_smooth(
                                 self.x, self.y,
                                 smoother_cls=supersmoother.SuperSmoother)
        pylab.plot(self.x, my_smoother.smooth_result, label='Supersmoother')
        finish_plot()

    @unittest.skip('Plots stuff')
    def test_supersmoother_bass(self):
        for bass in range(0, 1, 1):
            smoother = supersmoother.SuperSmoother()
            smoother._bass_enhancement = bass
            smoother.specify_data_set(self.x, self.y)
            smoother.compute()
            pylab.plot(self.x,
                       smoother._smoothed_best_spans.smooth_result,
                       '.',
                       label='Bass = {0}'.format(bass))
            # pylab.plot(self.x, smoother.smooth_result, label='Bass = {0}'.format(bass))
        finish_plot()

    @unittest.skip('long running')
    def test_average_best_span(self):
        N = 200
        pylab.figure()
        num_trials = 400
        avg = numpy.zeros(N)
        for i in range(num_trials):
            x, y = build_test_problem(N)
            my_smoother = smoother.perform_smooth(
                                 x, y,
                                 smoother_cls=supersmoother.SuperSmoother)
            avg += my_smoother._smoothed_best_spans.smooth_result
            if not (i + 1) % 20:
                print i + 1
        avg /= num_trials
        pylab.plot(self.x, avg)
        finish_plot()

    def test_known_curve(self):
        N = 100
        x = numpy.linspace(-1, 1, N)
        y = numpy.sin(4 * x)
        smooth = smoother.perform_smooth(x, y, smoother_cls=supersmoother.SuperSmoother)
        pylab.plot(x, smooth.smooth_result)
        smoother.DEFAULT_BASIC_SMOOTHER = smoother.BasicFixedSpanSmoother
        smooth = smoother.perform_smooth(x, y, smoother_cls=supersmoother.SuperSmoother)
        pylab.plot(x, smooth.smooth_result)
        pylab.plot(x, y, '.')
        pylab.show()

def finish_plot():
    pylab.legend()
    pylab.grid(color='0.7')
    pylab.xlabel('x')
    pylab.ylabel('y')
    pylab.show()
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
