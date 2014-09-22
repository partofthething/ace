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

import numpy
import numpy.linalg

from .supersmoother import SuperSmoother
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

    def solve(self):
        self._initialize()
        iters = 1
        while self._outer_error_is_decreasing() and iters < 10:
            print('* Starting outer iteration {0:03d}. Current err = {1:12.5E}'
                  ''.format(iters, self._last_outer_error))
            self._iterate_to_update_x_transforms()
            self._update_y_transform()
            iters += 1

    def _initialize(self):
        self._N = len(self._y)
        self._y_transform = self._y / numpy.linalg.norm(self._y)
        self._x_transforms = [numpy.zeros(self._N) for xi in self._x]

    def _outer_error_is_decreasing(self):
        is_decreasing, self._last_outer_error = self._error_is_decreasing(self._last_outer_error)
        return is_decreasing

    def _inner_error_is_decreasing(self):
        is_decreasing, self._last_inner_error = self._error_is_decreasing(self._last_inner_error)
        return is_decreasing

    def _error_is_decreasing(self, last_error):
        current_error = self._compute_error()
        if current_error <= last_error:
            is_decreasing = True
        else:
            is_decreasing = False
        return is_decreasing, current_error

    def _compute_error(self):
        sum_x = sum(self._x_transforms)
        err = sum((self._y_transform - sum_x) ** 2) / len(sum_x)
        return err

    def _iterate_to_update_x_transforms(self):
        iters = 1
        self._last_inner_error = float('inf')
        while self._inner_error_is_decreasing() and iters < 10:
            print('  Starting inner iteration {0:03d}. Current err = {1:12.5E}'
                  ''.format(iters, self._last_inner_error))
            self._update_x_transforms()
            iters += 1

    def _update_x_transforms(self):
        """
        Compute a new set of x-transform functions phi. 
        
        This is the first of the eponymous conditional expectations. The conditional
        expectations are computed using the SuperSmoother.  
        """
        for xtransform_index in range(len(self._x_transforms)):
            other_transforms = [transform for (k, transform) in enumerate(self._x_transforms)
                               if k != xtransform_index]
            updated_x_transform_choppy = numpy.zeros(self._N)
            for i in range(self._N):
                sum_of_others = sum([x_transform[i] for x_transform in other_transforms])
                updated_x_transform_choppy[i] = self._y_transform[i] - sum_of_others

            updated_x_transform_smooth = perform_smooth(self._x[xtransform_index],
                                                        updated_x_transform_choppy,
                                                        smoother_class=SuperSmoother)
            # updated_x_transform_smooth.plot()
            self._x_transforms[xtransform_index] = updated_x_transform_smooth.smooth_result

    def _update_y_transform(self):
        """
        Update the y-transform (theta).
        
        This uses the second conditional expectation
        """
        sum_of_x_transformations_choppy = sum(self._x_transforms)
        smooth = perform_smooth(self._y, sum_of_x_transformations_choppy,
                                smoother_class=SuperSmoother)
        sum_of_x_transformations_smooth = smooth.smooth_result
        self._y_transform = (sum_of_x_transformations_smooth /
                             numpy.linalg.norm(sum_of_x_transformations_smooth))

