"""Problem demonstrating supersmoother from [Friedman82]_."""

import math

import numpy.random
import matplotlib.pyplot as plt

from ace import smoother


def build_sample_smoother_problem_friedman82(N=200):
    """Sample problem from supersmoother publication."""
    x = numpy.random.uniform(size=N)
    err = numpy.random.standard_normal(N)
    y = numpy.sin(2 * math.pi * (1 - x) ** 2) + x * err
    return x, y

def run_friedman82_basic():
    """Run Friedman's test of fixed-span smoothers from Figure 2b."""
    x, y = build_sample_smoother_problem_friedman82()
    plt.figure()
    # plt.plot(x, y, '.', label='Data')
    for span in smoother.DEFAULT_SPANS:
        smooth = smoother.BasicFixedSpanSmoother()
        smooth.specify_data_set(x, y, sort_data=True)
        smooth.set_span(span)
        smooth.compute()
        plt.plot(x, smooth.smooth_result, '.', label='span = {0}'.format(span))
    plt.legend(loc='upper left')
    plt.grid(color='0.7')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Demo of fixed-span smoothers from Friedman 82')
    plt.savefig('sample_friedman82.png')

    return smooth

if __name__ == '__main__':
    run_friedman82_basic()
