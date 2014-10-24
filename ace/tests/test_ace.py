'''
Created on Sep 21, 2014

@author: nick
'''
from matplotlib import pyplot as plt
import matplotlib

import unittest

from ace import ace
import ace.smoother_diagnostics
import ace.supersmoother
import ace.validation.sample_problems as sample_problems
import ace.validation.validate_smoothers

class TestAce(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass
    @unittest.skip('skip')
    def test_sample_problem(self):

        x, y = sample_problems.sample_ace_problem_wang04(N=100)
        ace_solver = ace.ace.ACESolver()
        # ace_solver = ace.smoother_diagnostics.ACESolverWithPlots()
        # ace_solver._smoother_cls = ace.validation.validate_smoothers.SuperSmootherBreiman
        ace_solver._x = x
        ace_solver._y = y
        ace_solver.solve()
        plot_transforms(ace_solver, 'ace_results.png')

    @unittest.skip('yo')
    def test_sample_problem_supersmoother(self):
        ace_solver = build_sample_problem2(N=200, ace_cls=ace.smoother_diagnostics.ACESolverWithPlots)
        x = ace_solver._x[0]
        y = ace_solver._y
        # plt.figure()
        for bass in [5]:  # range(0, 10, 3):
            # smoother = ace.supersmoother.SuperSmoother()
            # for smoCls in [ace.smoother.BasicFixedSpanSmootherSlowUpdate, ace.smoother.BasicFixedSpanSmoother]:
            for smoCls in [ace.smoother.BasicFixedSpanSmoother]:
                smoother = smoCls()
                smoother.set_span(ace.smoother.BASS_SPAN)
                smoother.specify_data_set(x, y)
                smoother._bass_enhancement = bass
                smoother.compute()
                smoother.plot()
#             plt.plot(x,
#                     smoother.smooth_result,
#                     '.',
#                     label='Bass = {}'.format(bass))
        # plt.legend()
        # plt.show()
        # plt.close()

    #@unittest.skip('skip')
    def test_sample_problem2(self):
        x, y = ace.validation.sample_problems.sample_ace_problem_breiman85(400)
        plt.figure()
        plt.plot(x[0], y, '.')
        plt.savefig('sample_problem_data.png')
        ace_solver = ace.ace.ACESolver()
        # ace_solver = ace.smoother_diagnostics.ACESolverWithPlots()
        ace_solver._smoother_cls = ace.supersmoother.SuperSmoother
        #ace_solver._smoother_cls = ace.supersmoother.SuperSmootherWithPlots
        ace_solver._x = x
        ace_solver._y = y
        ace_solver.solve()
        plot_transforms(ace_solver, 'ace_results2.png')


def plot_transforms(ace_model, fName):
    matplotlib.rcParams.update({'font.size': 8})
    plt.figure()
    numCols = len(ace_model._x) / 2 + 1
    for i in range(len(ace_model._x)):
        plt.subplot(2, numCols, i + 1)
        plt.plot(ace_model._x[i], ace_model._x_transforms[i], '.', label='Phi {0}'.format(i))
    plt.subplot(2, numCols, i + 2)
    plt.plot(ace_model._y, ace_model._y_transform, '.', label='Theta')
    plt.legend()
    plt.savefig(fName)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
