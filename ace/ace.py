'''
The Alternating Condtional Expectation (ACE) algorithm

ACE was invented by L. Breiman and J. Friedman [1]. It is a powerful 
way to perform multidimensional regression without assuming
any functional form of the model. Given a data set:

    y = f(X)
    
where X is made up of a number of independent variables xi, ACE 
will tell you how y varies vs. each of the individual independents xi. 
This can be used to:

    a) Understand the relative shape and magnitude of y's dependence on each xi
    b) Produce a lightweight surrogate model of a more complex response
    c) other stuff 

ACE is available from Friedman's webpage as a FORTRAN program. This
same program has also been made available to the statistical language R in 
the form of the acepack module. This package represents a pure-Python version 
of ACE. It will allow people to understand how ACE works and will make it
easier to use ACE from other Python programs. 

[1] L. Breiman and J. Friedman, "Estimating Optimal Transformations
    for Multiple Regression and Correlation," Journal of the American Statistical 
    Association, Vol. 80, No. 391 (1985).
'''
import math

import numpy
import matplotlib.pyplot as plt

from .supersmoother import SuperSmoother, SuperSmootherWithPlots
from .smoother import perform_smooth

class ACESolver(object):
    '''
    Runs the Alternating Conditional Expectation algorithm to perform regressions

    '''

    def __init__(self):
        self._last_inner_error = float('inf')
        self._last_outer_error = float('inf')
        self._x = []
        self._y = None
        self._x_transforms = None
        self._y_transform = None
        self._smoother_cls = SuperSmoother

    def solve(self):
        self._initialize()
        self._outer_iters = 0
        while (self._outer_error_is_decreasing() or self._outer_iters < 10) and self._outer_iters < 25:
            print('* Starting outer iteration {0:03d}. Current err = {1:12.5E}'
                  ''.format(self._outer_iters, self._last_outer_error))
            self._iterate_to_update_x_transforms()
            self._update_y_transform()
            self._outer_iters += 1

    def _initialize(self):
        self._N = len(self._y)
        self._y_transform = self._y / self._norm(self._y)
        self._x_transforms = [numpy.zeros(self._N) for xi in self._x]

    def _norm(self, values):
        return numpy.linalg.norm(values)
        avg = sum(values) / float(len(values))
        var = (values - avg) ** 2
        return math.sqrt(sum(var))

    def _outer_error_is_decreasing(self):
        is_decreasing, self._last_outer_error = self._error_is_decreasing(self._last_outer_error)
        return is_decreasing

    def _inner_error_is_decreasing(self):
        is_decreasing, self._last_inner_error = self._error_is_decreasing(self._last_inner_error)
        return is_decreasing

    def _error_is_decreasing(self, last_error):
        current_error = self._compute_error()
        if current_error < last_error:
            is_decreasing = True
        else:
            is_decreasing = False
        return is_decreasing, current_error

    def _compute_error(self):
        sum_x = sum(self._x_transforms)
        err = sum((self._y_transform - sum_x) ** 2) / len(sum_x)
        return err

    def _iterate_to_update_x_transforms(self):
        self._inner_iters = 0
        self._last_inner_error = float('inf')
        while self._inner_error_is_decreasing():
            print('  Starting inner iteration {0:03d}. Current err = {1:12.5E}'
                  ''.format(self._inner_iters, self._last_inner_error))
            self._update_x_transforms()
            self._inner_iters += 1

    def _update_x_transforms(self):
        """
        Compute a new set of x-transform functions phi. 
        
        This is the first of the eponymous conditional expectations. The conditional
        expectations are computed using the SuperSmoother.  
        """
        for xtransform_index in range(len(self._x_transforms)):
            xk = self._x[xtransform_index]

            other_transforms = [transform
                                for (k, transform) in enumerate(self._x_transforms)
                                if k != xtransform_index]
            if other_transforms:
                sum_of_others = numpy.sum(other_transforms, axis=0)
            else:
                sum_of_others = numpy.zeros(len(self._y))

            updated_x_transform = self._y_transform - sum_of_others
            updated_x_transform_smooth = perform_smooth(xk, updated_x_transform,
                                                        smoother_cls=self._smoother_cls).smooth_result
            plt.figure()
            plt.plot(xk, self._y_transform, '.', label='theta')
            # plt.plot(self._x[xtransform_index], sum_of_others, '.', label='others')
            plt.plot(xk, updated_x_transform_smooth, '.', label='newPhi')
            plt.legend()
            plt.xlabel('x_{0}'.format(xtransform_index))
            plt.ylabel('phi_{0}'.format(xtransform_index))
            plt.savefig('updated_x_transform{0:03d}_{1:03d}.png'.format(self._outer_iters, self._inner_iters))
            plt.close()
            self._x_transforms[xtransform_index] = updated_x_transform_smooth


    def _update_y_transform(self):
        """
        Update the y-transform (theta).
        
        This uses the second conditional expectation
        """
        sum_of_x_transformations_choppy = numpy.sum(self._x_transforms, axis=0)
        smooth = perform_smooth(self._y, sum_of_x_transformations_choppy,
                                smoother_cls=self._smoother_cls)
        sum_of_x_transformations_smooth = smooth.smooth_result
        smooth.plot('theta-smooth.png')
        self._y_transform = (sum_of_x_transformations_smooth /
                             self._norm(sum_of_x_transformations_smooth))

