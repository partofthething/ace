"""Run the Sample problem from [Wang04]_."""

import numpy

from ace import ace

def build_sample_ace_problem_wang04(N=100):
    """Build sample problem from Wang 2004."""
    x = [numpy.random.uniform(-1, 1, size=N)
         for _i in range(0, 5)]
    noise = numpy.random.standard_normal(N)
    y = numpy.log(4.0 + numpy.sin(4 * x[0]) + numpy.abs(x[1]) + x[2] ** 2 +
                  x[3] ** 3 + x[4] + 0.1 * noise)
    return x, y

def run_wang04():
    """Run sample problem."""
    x, y = build_sample_ace_problem_wang04(N=200)
    ace_solver = ace.ACESolver()
    ace_solver.specify_data_set(x, y)
    ace_solver.solve()
    try:
        ace.plot_transforms(ace_solver, 'ace_transforms_wang04.png')
        ace.plot_input(ace_solver, 'ace_input_wang04.png')
    except ImportError:
        pass

    return ace_solver

if __name__ == '__main__':
    run_wang04()
