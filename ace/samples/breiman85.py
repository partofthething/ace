'''
Run the Sample ACE problem from [Breiman85]_
'''

import numpy
import numpy.random
import scipy.special

from ace import ace

numpy.random.seed(9287349087)

def build_sample_ace_problem_breiman85(N=200):
    """
    Sample problem from Breiman 1985
    """
    x3 = numpy.random.standard_normal(N)
    x = scipy.special.cbrt(x3)
    noise = numpy.random.standard_normal(N)
    y = numpy.exp((x ** 3.0) + noise)
    return [x], y


def run_breiman85():
    x, y = build_sample_ace_problem_breiman85(200)
    ace_solver = ace.ACESolver()
    ace_solver.specify_data_set(x, y)
    ace_solver.solve()
    try:
        ace.plot_transforms(ace_solver, 'sample_ace_breiman85.png')
    except ImportError:
        pass

    return ace_solver

if __name__ == '__main__':
    run_breiman85()