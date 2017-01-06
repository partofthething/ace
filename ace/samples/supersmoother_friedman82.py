"""Problem demonstrating supersmoother from [Friedman82]_."""

import matplotlib.pyplot as plt

from ace.samples import smoother_friedman82
from ace.supersmoother import SuperSmoother

def run_friedman82_super():
    """Run Friedman's test of fixed-span smoothers from Figure 2b."""
    x, y = smoother_friedman82.build_sample_smoother_problem_friedman82()
    plt.figure()
    smooth = SuperSmoother()
    smooth.specify_data_set(x, y, sort_data=True)
    smooth.compute()
    plt.plot(x, y, '.', label='Data')
    plt.plot(smooth.x, smooth.smooth_result, 'o', label='Smooth')
    plt.grid(color='0.7')
    plt.legend(loc='upper left')
    plt.title('Demo of SuperSmoother based on Friedman 82')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('sample_supersmoother_friedman82.png')

    return smooth

if __name__ == '__main__':
    run_friedman82_super()
