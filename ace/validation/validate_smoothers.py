"""
A few validation problems to make sure the smoothers are working as expected.

These depend on the supsmu module, which was created using f2py from Breiman's supsmu.f
"""

import matplotlib.pyplot as plt
import numpy

from ace.samples import smoother_friedman82
import ace.smoother as smoother
import ace.supersmoother as supersmoother

# pylint: disable=protected-access, missing-docstring

try:
    import mace
except ImportError:
    print("WARNING: An F2Pyd version of Breiman's supsmu is not available. "
          "Validations will not work")
    raise

def validate_basic_smoother():
    """Run Friedman's test from Figure 2b."""
    x, y = sort_data(*smoother_friedman82.build_sample_smoother_problem_friedman82())
    plt.figure()
    # plt.plot(x, y, '.', label='Data')
    for span in smoother.DEFAULT_SPANS:
        my_smoother = smoother.perform_smooth(x, y, span)
        friedman_smooth, _resids = run_friedman_smooth(x, y, span)
        plt.plot(x, my_smoother.smooth_result, '.-', label='pyace span = {0}'.format(span))
        plt.plot(x, friedman_smooth, '.-', label='Friedman span = {0}'.format(span))
    finish_plot()


def validate_basic_smoother_resid():
    """Compare residuals."""
    x, y = sort_data(*smoother_friedman82.build_sample_smoother_problem_friedman82())
    plt.figure()
    for span in smoother.DEFAULT_SPANS:
        my_smoother = smoother.perform_smooth(x, y, span)
        _friedman_smooth, resids = run_friedman_smooth(x, y, span)  # pylint: disable=unused-variable
        plt.plot(x, my_smoother.cross_validated_residual, '.-',
                 label='pyace span = {0}'.format(span))
        plt.plot(x, resids, '.-', label='Friedman span = {0}'.format(span))
    finish_plot()

def validate_supersmoother():
    """Validate the supersmoother."""
    x, y = smoother_friedman82.build_sample_smoother_problem_friedman82()
    x, y = sort_data(x, y)
    my_smoother = smoother.perform_smooth(x, y, smoother_cls=supersmoother.SuperSmootherWithPlots)
    # smoother.DEFAULT_BASIC_SMOOTHER = BasicFixedSpanSmootherBreiman
    supsmu_result = run_freidman_supsmu(x, y, bass_enhancement=0.0)
    mace_result = run_mace_smothr(x, y, bass_enhancement=0.0)
    plt.plot(x, y, '.', label='Data')
    plt.plot(x, my_smoother.smooth_result, '-', label='pyace')
    plt.plot(x, supsmu_result, '--', label='SUPSMU')
    plt.plot(x, mace_result, ':', label='SMOOTH')
    plt.legend()
    plt.savefig('supersmoother_validation.png')

def validate_supersmoother_bass():
    """Validate the supersmoother with extra bass."""
    x, y = smoother_friedman82.build_sample_smoother_problem_friedman82()
    plt.figure()
    plt.plot(x, y, '.', label='Data')
    for bass in range(0, 10, 3):
        smooth = supersmoother.SuperSmoother()
        smooth.set_bass_enhancement(bass)
        smooth.specify_data_set(x, y)
        smooth.compute()
        plt.plot(x, smooth.smooth_result, '.', label='Bass = {0}'.format(bass))
        # pylab.plot(self.x, smoother.smooth_result, label='Bass = {0}'.format(bass))
    finish_plot()

def validate_average_best_span():
    """Figure 2d? from Friedman."""
    N = 200
    num_trials = 400
    avg = numpy.zeros(N)
    for i in range(num_trials):
        x, y = smoother_friedman82.build_sample_smoother_problem_friedman82(N=N)
        my_smoother = smoother.perform_smooth(
            x, y, smoother_cls=supersmoother.SuperSmoother
        )
        avg += my_smoother._smoothed_best_spans.smooth_result
        if not (i + 1) % 20:
            print(i + 1)
    avg /= num_trials
    plt.plot(my_smoother.x, avg, '.', label='Average JCV')
    finish_plot()

def validate_known_curve():
    """Validate on a sin function."""
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
    """Help with plotting."""
    plt.legend()
    plt.grid(color='0.7')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def run_freidman_supsmu(x, y, bass_enhancement=0.0):
    """Run the FORTRAN supersmoother."""
    N = len(x)
    weight = numpy.ones(N)
    results = numpy.zeros(N)
    flags = numpy.zeros((N, 7))
    mace.supsmu(x, y, weight, 1, 0.0, bass_enhancement, results, flags)
    return results

def run_friedman_smooth(x, y, span):
    """Run the FORTRAN smoother."""
    N = len(x)
    weight = numpy.ones(N)
    results = numpy.zeros(N)
    residuals = numpy.zeros(N)
    mace.smooth(x, y, weight, span, 1, 1e-7, results, residuals)
    return results, residuals

def run_mace_smothr(x, y, bass_enhancement=0.0):  # pylint: disable=unused-argument
    """Run the FORTRAN SMOTHR."""
    N = len(x)
    weight = numpy.ones(N)
    results = numpy.zeros(N)
    flags = numpy.zeros((N, 7))
    mace.smothr(1, x, y, weight, results, flags)
    return results

class BasicFixedSpanSmootherBreiman(smoother.Smoother):
    """Runs FORTRAN Smooth."""

    def compute(self):
        """Run smoother."""
        self.smooth_result, self.cross_validated_residual = run_friedman_smooth(
            self.x, self.y, self._span
        )

class SuperSmootherBreiman(smoother.Smoother):
    """Run FORTRAN Supersmoother."""

    def compute(self):
        """Run SuperSmoother."""
        self.smooth_result = run_freidman_supsmu(self.x, self.y)
        self._store_unsorted_results(self.smooth_result, numpy.zeros(len(self.smooth_result)))

def sort_data(x, y):
    """Sort the data."""
    xy = sorted(zip(x, y))
    x, y = zip(*xy)
    return x, y

if __name__ == '__main__':
    validate_basic_smoother()
    # validate_basic_smoother_resid()
    #validate_supersmoother()
