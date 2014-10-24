"""
Collection of sample problems useful for testing ACE, supersmoother
"""
import math

import numpy
import numpy.random
import scipy.special

from ace.validation.validate_smoothers import sort_data

def sample_ace_problem_wang04(N=100):
    """
    Sample problem from Wang 2004
    """

    x = [numpy.random.uniform(-1, 1, size=N)
         for _i in range(0, 5)]
    noise = numpy.random.standard_normal(N)
    y = numpy.log(4.0 + numpy.sin(4 * x[0]) + numpy.abs(x[1]) + x[2] ** 2 +
                 x[3] ** 3 + x[4] + 0.1 * noise)
    return x, y

def sample_ace_problem_breiman85(N=200):
    """
    Sample problem 1 from Breiman 1985
    """
    x3 = numpy.random.standard_normal(N)
    x = scipy.special.cbrt(x3)
    noise = numpy.random.standard_normal(N)
    y = numpy.exp((x**3.0) + noise)
    #x, y = sort_data(x, y)
    return [x], y

def sample_smoother_problem_brieman82(N=200):
    """
    Sample problem from supersmoother pub.
    """
    x = numpy.random.uniform(size=N)
    err = numpy.random.standard_normal(N)
    y = numpy.sin(2 * math.pi * (1 - x) ** 2) + x * err
    return x, y
