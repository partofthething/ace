'''
Created on Sep 21, 2014

@author: nick
'''
import unittest
import random

import numpy
import matplotlib
from matplotlib import pyplot as plt

from ace import ace

class TestAce(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_sample_problem(self):

        ace_solver = build_sample_problem()
        ace_solver.solve()
        plot_transforms(ace_solver)


def plot_transforms(ace_model):
    matplotlib.rcParams.update({'font.size': 8})
    plt.figure()
    for i in range(5):
        plt.subplot(2, 3, i + 1)
        plt.plot(ace_model._x[i], ace_model._x_transforms[i], '.', label='Phi {0}'.format(i))
    plt.subplot(2, 3, 6)
    plt.plot(ace_model._y, ace_model._y_transform, '.', label='Theta')
    plt.legend()
    plt.savefig('ace_results.png')

def build_sample_problem(ace_cls=None):
    N = 100

    x = [numpy.array([random.random() * 2.0 - 1.0 for i in range(N)])
         for _i in range(0, 5)]
    noise = numpy.random.standard_normal(N)
    y = numpy.log(4.0 + numpy.sin(4 * x[0]) + numpy.abs(x[1]) + x[2] ** 2 +
                 x[3] ** 3 + x[4] + 0.1 * noise)
    if ace_cls is None:
        ace_cls = ace.ACESolver
    ace_solver = ace_cls()
    ace_solver._x = x
    ace_solver._y = y

    return ace_solver

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
