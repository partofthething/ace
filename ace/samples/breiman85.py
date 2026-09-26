"""Run the Sample ACE problem from [Breiman85]_."""

import numpy.random
import scipy.special

from ace import ace


def build_sample_ace_problem_breiman85(N=200):
    """Sample problem from Breiman 1985."""
    x_cubed = numpy.random.standard_normal(N)
    x = scipy.special.cbrt(x_cubed)
    noise = numpy.random.standard_normal(N)
    y = numpy.exp((x**3.0) + noise)
    return [x], y


def build_sample_ace_problem_breiman2(N=200):
    """Build sample problem y = exp(sin(2 pi x) + noise/2) from Breiman 1985."""
    x = numpy.random.uniform(0, 1, size=N)
    noise = numpy.random.standard_normal(N)
    y = numpy.exp(numpy.sin(2 * numpy.pi * x) + noise / 2.0)
    return [x], y


def run_breiman85():
    """Run Breiman 85 sample."""
    x, y = build_sample_ace_problem_breiman85(200)
    ace_solver = ace.ACESolver()
    ace_solver.specify_data_set(x, y)
    ace_solver.solve()
    try:
        ace.plot_transforms(ace_solver, "sample_ace_breiman85.png")
    except ImportError:
        pass
    return ace_solver


def run_breiman2():
    """Run Breiman's other sample problem."""
    x, y = build_sample_ace_problem_breiman2(200)
    ace_solver = ace.ACESolver()
    ace_solver.specify_data_set(x, y)
    ace_solver.solve()
    try:
        plt = ace.plot_transforms(ace_solver, None)
    except ImportError:
        return ace_solver

    # log(y) and sin(2 pi x) are close to optimal. Scale them like theta for comparison.
    log_y = numpy.log(y)
    offset, scale = numpy.mean(log_y), numpy.std(log_y)
    x_sorted = numpy.sort(x[0])
    plt.subplot(1, 2, 1)
    plt.plot(x_sorted, numpy.sin(2.0 * numpy.pi * x_sorted) / scale, label="analytic")
    plt.legend()
    plt.subplot(1, 2, 2)
    y_sorted = numpy.sort(y)
    plt.plot(y_sorted, (numpy.log(y_sorted) - offset) / scale, label="analytic")
    plt.legend(loc="lower right")
    plt.savefig("sample_ace_breiman85_2.png")

    return ace_solver


if __name__ == "__main__":
    run_breiman2()
