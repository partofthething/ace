"""Run the Sample ACE problem from [Breiman85]_."""

import numpy.random
import scipy.special

from ace import ace


def build_sample_ace_problem_breiman85(N=200):
    """Sample problem from Breiman 1985."""
    x_cubed = numpy.random.standard_normal(N)
    x = scipy.special.cbrt(x_cubed)
    noise = numpy.random.standard_normal(N)
    y = numpy.exp((x ** 3.0) + noise)
    return [x], y


def build_sample_ace_problem_breiman2(N=500):
    """Build sample problem y(x) = exp(sin(x))."""
    x = numpy.linspace(0, 1, N)
    # x = numpy.random.uniform(0, 1, size=N)
    noise = numpy.random.standard_normal(N)
    y = numpy.exp(numpy.sin(2 * numpy.pi * x)) + 0.0 * noise
    return [x], y


def run_breiman85():
    """Run Breiman 85 sample."""
    x, y = build_sample_ace_problem_breiman85(200)
    ace_solver = ace.ACESolver()
    ace_solver.specify_data_set(x, y)
    ace_solver.solve()
    try:
        ace.plot_transforms(ace_solver, 'sample_ace_breiman85.png')
    except ImportError:
        pass
    return ace_solver

def run_breiman2():
    """Run Breiman's other sample problem."""
    x, y = build_sample_ace_problem_breiman2(500)
    ace_solver = ace.ACESolver()
    ace_solver.specify_data_set(x, y)
    ace_solver.solve()
    try:
        plt = ace.plot_transforms(ace_solver, None)
    except ImportError:
        pass

    plt.subplot(1, 2, 1)
    phi = numpy.sin(2.0 * numpy.pi * x[0])
    plt.plot(x[0], phi, label='analytic')
    plt.legend()
    plt.subplot(1, 2, 2)
    y = numpy.exp(phi)
    plt.plot(y, phi, label='analytic')
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig('no_noise_linear_x.png')

    return ace_solver


if __name__ == '__main__':
    run_breiman2()
