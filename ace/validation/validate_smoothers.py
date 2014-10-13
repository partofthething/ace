'''
A few validation problems to make sure the smoothers are working as expected

These depend on the supsmu module, which was created using f2py from Breiman's supsmu.f
'''

from matplotlib import pyplot as plt
import numpy

import sample_problems
import ace.smoother as smoother
import ace.supersmoother as supersmoother
import supsmu

def validate_basic_smoother():
    """
    Runs Friedman's test from Figure 2b. 
    """
    x, y = sort_data(*sample_problems.sample_smoother_problem_brieman82())
    plt.figure()
    # plt.plot(x, y, '.', label='Data')
    for span in smoother.DEFAULT_SPANS:
        my_smoother = smoother.perform_smooth(x, y, span)
        friedman_smooth, resids = run_friedman_smooth(x, y, span)
        plt.plot(x, my_smoother.smooth_result, '.-', label='pyace span = {0}'.format(span))
        plt.plot(x, friedman_smooth, '.-', label='Friedman span = {0}'.format(span))
    finish_plot()


def validate_basic_smoother_resid():
    """
    compare residuals
    """
    x, y = sort_data(*sample_problems.sample_smoother_problem_brieman82())
    plt.figure()
    for span in smoother.DEFAULT_SPANS:
        my_smoother = smoother.perform_smooth(x, y, span)
        friedman_smooth, resids = run_friedman_smooth(x, y, span)
        plt.plot(x, my_smoother.cross_validated_residual, '.-', label='pyace span = {0}'.format(span))
        plt.plot(x, resids, '.-', label='Friedman span = {0}'.format(span))
    finish_plot()

def validate_supersmoother():
    x, y = sample_problems.sample_smoother_problem_brieman82()
    x, y = sort_data(x, y)
    my_smoother = smoother.perform_smooth(x, y, smoother_cls=supersmoother.SuperSmootherWithPlots)
    # smoother.DEFAULT_BASIC_SMOOTHER = BasicFixedSpanSmootherBreiman
    supsmu_result = run_freidman_supsmu(x, y)
    plt.plot(x, y, '.', label='Data')
    plt.plot(x, my_smoother.smooth_result, '-', label='pyace')
    plt.plot(x, supsmu_result, '-', label='SUPSMU')
    plt.legend()
    plt.show()

def validate_supersmoother_bass():
    x, y = sample_problems.sample_smoother_problem_brieman82()
    plt.figure()
    plt.plot(x, y, '.', label='Data')
    for bass in range(0, 10, 3):
        smoother = supersmoother.SuperSmoother()
        smoother._bass_enhancement = bass
        smoother.specify_data_set(x, y)
        smoother.compute()
        plt.plot(x,
                   smoother.smooth_result,
                   '.',
                   label='Bass = {0}'.format(bass))
        # pylab.plot(self.x, smoother.smooth_result, label='Bass = {0}'.format(bass))
    finish_plot()


def validate_average_best_span(self):
    """
    Figure 2d? from Friedman
    """
    N = 200
    num_trials = 400
    avg = numpy.zeros(N)
    for i in range(num_trials):
        x, y = sample_problems.sample_smoother_problem_brieman82(N=N)
        my_smoother = smoother.perform_smooth(
                             x, y,
                             smoother_cls=supersmoother.SuperSmoother)
        avg += my_smoother._smoothed_best_spans.smooth_result
        if not (i + 1) % 20:
            print(i + 1)
    avg /= num_trials
    plt.plot(my_smoother._x, avg, '.', label='Average JCV')
    finish_plot()


def validate_known_curve(self):
    plt.figure()
    N = 100
    x = numpy.linspace(-1, 1, N)
    y = numpy.sin(4 * x)
    smoother.DEFAULT_BASIC_SMOOTHER = smoother.BasicFixedSpanSmootherSlowUpdate
    smooth = smoother.perform_smooth(x, y, smoother_cls=supersmoother.SuperSmoother)
    plt.plot(x, smooth.smooth_result, label='Slow')
    smoother.DEFAULT_BASIC_SMOOTHER = smoother.BasicFixedSpanSmoother
    smooth = smoother.perform_smooth(x, y, smoother_cls=supersmoother.SuperSmoother)
    plt.plot(x, smooth.smooth_result, label='Fast')
    plt.plot(x, y, '.', label='data')
    plt.legend()
    plt.show()

def finish_plot():
    plt.legend()
    plt.grid(color='0.7')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def run_freidman_supsmu(x, y):
    N = len(x)
    weight = numpy.ones(N)
    results = numpy.zeros(N)
    sc = numpy.zeros((N, 7))
    bass_enhancement = 0.0
    supsmu.supsmu(x, y, weight, 1, 0.0, bass_enhancement, results, sc)
    return results

def run_friedman_smooth(x, y, span):
    N = len(x)
    weight = numpy.ones(N)
    results = numpy.zeros(N)
    residuals = numpy.zeros(N)
    supsmu.smooth(x, y, weight, span, 1, 1e-7, results, residuals)
    return results, residuals


class BasicFixedSpanSmootherBreiman(smoother.Smoother):
    """
    Runs FORTRAN Smooth
    """
    def compute(self):
        self.smooth_result, self.cross_validated_residual = run_friedman_smooth(self._x, self._y, self._span)

class SuperSmootherBreiman(smoother.Smoother):
    """
    Runs FORTRAN Supersmoother
    """
    def compute(self):
        self.smooth_result = run_freidman_supsmu(self._x, self._y)
        self._store_unsorted_results(self.smooth_result, numpy.zeros(len(self.smooth_result)))

def sort_data(x, y):
    xy = zip(x, y)
    xy.sort()
    x, y = zip(*xy)
    return x, y

if __name__ == '__main__':
    # validate_basic_smoother()
    # validate_basic_smoother_resid()
    validate_supersmoother()
