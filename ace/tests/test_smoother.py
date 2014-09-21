'''
Smoother tests

Note: these aren't real unittests yet. Just making sure we can reproduce 
the plots from the paper. 

'''
import math
import unittest

import numpy
import pylab

import ace.smoother
import ace.supersmoother

class TestSmoothers(unittest.TestCase):

    def setUp(self):
        N = 200
        self.x = numpy.linspace(0, 1, N)
        # add iid standard normal error
        err = numpy.random.standard_normal(N)
        self.y = [math.sin(2 * math.pi * (1 - xi) ** 2) + xi * ei for xi, ei in zip(self.x, err)]
        pylab.figure()
        pylab.plot(self.x, self.y, '.', label='Data')

    def test_basic_smoother(self):
        """
        Runs Friedman's test from Figure 2b. 
        """

        for span in ace.smoother.DEFAULT_SPANS:
            smoother = ace.smoother.perform_basic_smooth(self.x, self.y, span)
            pylab.plot(self.x, smoother.smooth_result, label='Span = {0}'.format(span))
        finish_plot()

    def test_supersmoother(self):
        smoother = ace.smoother.perform_basic_smooth(
                                 self.x, self.y,
                                 smoother_class=ace.supersmoother.SuperSmoother)
        pylab.plot(self.x, smoother.smooth_result, label='Supersmoother')
        finish_plot()


def finish_plot():
    pylab.legend()
    pylab.grid(color='0.7')
    pylab.xlabel('x')
    pylab.ylabel('y')
    pylab.show()
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
