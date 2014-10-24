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
        self._xi_sorted = None
        self._yi_sorted = None
        self._x_transforms = None
        self._y_transform = None
        self._smoother_cls = SuperSmoother

    def solve(self):
        self._initialize()
        self._outer_iters = 0
        while self._outer_error_is_decreasing():
            print('* Starting outer iteration {0:03d}. Current err = {1:12.5E}'
                  ''.format(self._outer_iters, self._last_outer_error))
            self._iterate_to_update_x_transforms()
            self._update_y_transform()
            self._outer_iters += 1

    def _initialize(self):
        self._N = len(self._y)
        self._y_transform = self._y - numpy.mean(self._y)
        self._y_transform /= numpy.std(self._y_transform)
        self._x_transforms = [numpy.zeros(self._N) for xi in self._x]
        self._compute_sorted_indices()

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

    def _compute_sorted_indices(self):
        """
        The smoothers need sorted data. This sorts it from the perspective of each transform.

        We only have to sort the data once.  
        """
        sorted_indices = []
        for to_sort in [self._y] + self._x:
            data_w_indices = [(val, i) for (i, val) in enumerate(to_sort)]
            data_w_indices.sort()
            sorted_indices.append([i for val, i in data_w_indices])
        # save in meaningful variable names
        self._yi_sorted = sorted_indices[0]  # list (like self._y)
        self._xi_sorted = sorted_indices[1:] # list of lists (like self._x)
        

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
        
        phik(xk) = theta(y) - sum of phii(xi) over i!=k
        
        This is the first of the eponymous conditional expectations. The conditional
        expectations are computed using the SuperSmoother.  
        """
        
        theta_minus_phis = self._y_transform - numpy.sum(self._x_transforms, axis=0) # z5
        for xtransform_index in range(len(self._x_transforms)):
            xtransform = self._x_transforms[xtransform_index] # don't change iterator
            sorted_data_indices = self._xi_sorted[xtransform_index]
            xk_sorted = sort_vector(self._x[xtransform_index], sorted_data_indices)
            xtransform_sorted = sort_vector(xtransform, sorted_data_indices)
            theta_minus_phis_sorted = sort_vector(theta_minus_phis, sorted_data_indices)

            to_smooth = theta_minus_phis_sorted + xtransform_sorted  # z1 = z5 + phi          
            smoother = perform_smooth(xk_sorted, to_smooth, smoother_cls=self._smoother_cls)
            #smoother.plot('phi-transform{0}'.format(self._outer_iters))
            updated_x_transform_smooth = smoother.smooth_result
            updated_x_transform_smooth -= numpy.mean(updated_x_transform_smooth) # z3

            # store updated transform in the order of the original data
            unsorted_xt = unsort_vector(updated_x_transform_smooth, sorted_data_indices)
            self._x_transforms[xtransform_index] = unsorted_xt

            tmp_unsorted = unsort_vector(to_smooth, sorted_data_indices)
            theta_minus_phis = tmp_unsorted - unsorted_xt  # z5 = z1 - z3
    
        
    def _update_y_transform(self):
        """
        Update the y-transform (theta).
        
        This uses the second conditional expectation
        """
        # sort wrt y
        sorted_data_indices = self._yi_sorted  # e.g [4,2,1,0,3] if y was [30,10,5,1,25]
        sorted_xtransforms = []
        for xt in self._x_transforms:
            sorted_xt = sort_vector(xt,sorted_data_indices)
            sorted_xtransforms.append(sorted_xt)

        sum_of_x_transformations_choppy = numpy.sum(sorted_xtransforms, axis=0)
        y_sorted = sort_vector(self._y, sorted_data_indices)
        smooth = perform_smooth(y_sorted, sum_of_x_transformations_choppy,
                                smoother_cls=self._smoother_cls)
        sum_of_x_transformations_smooth = smooth.smooth_result
        #smooth.plot('theta-smooth{0}.png'.format(self._outer_iters))
        sum_of_x_transformations_smooth -= numpy.mean(sum_of_x_transformations_smooth)
        sum_of_x_transformations_smooth /= numpy.std(sum_of_x_transformations_smooth)
        # unsort to save in the original data
        self._y_transform = unsort_vector(sum_of_x_transformations_smooth, sorted_data_indices)

def sort_vector(data, indices_of_increasing):
    return numpy.array([data[i] for i in indices_of_increasing])

def unsort_vector(data, indices_of_increasing):
    return numpy.array([data[indices_of_increasing.index(i)] for i in range(len(data))])

