'''
Python port of J. Friedman's 1984 smoother

[1] J. Friedman, "A Variable Span Smoother", 1984 
        http://www.slac.stanford.edu/cgi-wrap/getdoc/slac-pub-3477.pdf
'''
import math

import numpy
import pylab

TWEETER_SPAN = 0.05
MID_SPAN = 0.2
BASS_SPAN = 0.5

DEFAULT_SPANS = (TWEETER_SPAN, MID_SPAN, BASS_SPAN)

class Smoother(object):
    '''
    Smoothers accept data and produce smoother curves that fit the data.
    '''

    def __init__(self):
        self._x = []
        self._y = []

        self._mean_x_in_window = 0.0
        self._mean_y_in_window = 0.0
        self._covariance_in_window = 0.0
        self._variance_in_window = 0.0

        self._span = None
        self.smooth_result = []
        self.cross_validated_residual = None
        self.window_size = None
        self._original_index_of_xvalue = []  # for dealing w/ unsorted data
        self._window_bound_lower = 0
        self._x_in_window = []
        self._y_in_window = []

    def add_data_point_xy(self, x, y):
        """
        add a new data point to the data set to be smoothed
        """
        self._x.append(x)
        self._y.append(y)

    def specify_data_set(self, x_input, y_input):
        """
        Fully define data by lists of x values and y values. 
        
        This will sort them by increasing x but remember how to unsort them for providing results. 

        Parameters
        ----------
        xValues : iterable
            list of floats that represent x
        yValues : iterable
            list of floats that represent y(x) for each x
        """

        xy = sorted(zip(x_input, y_input))
        x, y = zip(*xy)  # pylint: disable=star-args
        x_input_list = list(x_input)
        self._original_index_of_xvalue = [x_input_list.index(xi) for xi in x]
        if len(set(self._original_index_of_xvalue)) != len(x):
            raise RuntimeError('There are some non-unique original indices')

        self._x = x
        self._y = y

    def set_span(self, span):
        self._span = span

    def compute(self):
        raise NotImplementedError

    def plot(self, fName=None):
        pylab.figure()
        xy = zip(self._x, self.smooth_result)
        xy.sort()
        x, y = zip(*xy)
        pylab.plot(x, y, '-')
        # pylab.plot(self._x, self._y, '.')
        if fName:
            pylab.savefig(fName)
        else:
            pylab.show()
        pylab.close()

    def _store_unsorted_results(self, smooth, residual):
        """
        Convert sorted smooth/residual back to as-input order
        """
        self.smooth_result = numpy.zeros(len(self._y))
        self.cross_validated_residual = numpy.zeros(len(residual))
        original_x = numpy.zeros(len(self._y))
        for i, (xval, smooth_val, residual_val) in enumerate(zip(self._x, smooth, residual)):
            original_index = self._original_index_of_xvalue[i]
            original_x[original_index] = xval
            self.smooth_result[original_index] = smooth_val
            self.cross_validated_residual[original_index] = residual_val
        self._x = original_x


class BasicFixedSpanSmoother(Smoother):
    """
    A basic fixed-span smoother

    Simple least-squares linear local smoother. 
    
    Uses fast updates of means, variances. 
    """

    def compute(self):
        """
        Perform the smoothing operations
        """
        self._compute_window_size()
        smooth = []
        residual = []

        x, y = self._x, self._y

        # step through x and y data with a window window_size wide.
        self._update_values_in_window()
        self._update_mean_in_window()
        self._update_variance_in_window()
        for i, (xi, yi) in enumerate(zip(x, y)):
            if (i - self._neighbors_on_each_side) > 0.0 and (i + self._neighbors_on_each_side) < len(x):
                self._advance_window()
            smooth_here = self._compute_smooth_during_construction(xi)
            residual_here = self._compute_cross_validated_residual_here(xi, yi, smooth_here)
            smooth.append(smooth_here)
            residual.append(residual_here)

        self._store_unsorted_results(smooth, residual)

    def _compute_window_size(self):
        """
        Make a symmetric neighborhood with J/2 values on each side of current position j
        """
        self._neighbors_on_each_side = int(len(self._x) * self._span) / 2
        self.window_size = self._neighbors_on_each_side * 2 + 1

    def _update_values_in_window(self):
        window_bound_upper = self._window_bound_lower + self.window_size
        self._x_in_window = self._x[self._window_bound_lower:window_bound_upper]
        self._y_in_window = self._y[self._window_bound_lower:window_bound_upper]

    def _update_mean_in_window(self):
        self._mean_x_in_window = numpy.mean(self._x_in_window)
        self._mean_y_in_window = numpy.mean(self._y_in_window)

    def _update_variance_in_window(self):
        self._covariance_in_window = sum([(xj - self._mean_x_in_window) *
                                          (yj - self._mean_y_in_window)
                      for xj, yj in zip(self._x_in_window, self._y_in_window)])

        self._variance_in_window = sum([(xj - self._mean_x_in_window) ** 2 for xj
                                        in self._x_in_window])

    def _advance_window(self):
        """
        Update values in current window and the current window means and variances. 
        """
        x_to_remove, y_to_remove = self._x_in_window[0], self._y_in_window[0]

        self._window_bound_lower += 1
        self._update_values_in_window()
        x_to_add, y_to_add = self._x_in_window [-1], self._y_in_window[-1]

        self._remove_observation(x_to_remove, y_to_remove)
        self._add_observation(x_to_add, y_to_add)

    def _remove_observation(self, x_to_remove, y_to_remove):
        self._remove_observation_to_variances(x_to_remove, y_to_remove)
        self._remove_observation_from_means(x_to_remove, y_to_remove)
        self.window_size -= 1

    def _add_observation(self, x_to_add, y_to_add):
        self._add_observation_to_means(x_to_add, y_to_add)
        self._add_observation_to_variances(x_to_add, y_to_add)
        self.window_size += 1

    def _add_observation_to_means(self, xj, yj):
        """
        Update the means without recalculating for the addition of one observation
        """
        self._mean_x_in_window = ((self.window_size * self._mean_x_in_window + xj) /
                                  (self.window_size + 1.0))
        self._mean_y_in_window = ((self.window_size * self._mean_y_in_window + yj) /
                                  (self.window_size + 1.0))

    def _remove_observation_from_means(self, xj, yj):
        """
        Update the means without recalculating for the deletion of one observation
        """
        self._mean_x_in_window = ((self.window_size * self._mean_x_in_window - xj) /
                                  (self.window_size - 1.0))
        self._mean_y_in_window = ((self.window_size * self._mean_y_in_window - yj) /
                                  (self.window_size - 1.0))

    def _add_observation_to_variances(self, xj, yj):
        """
        Quickly update the variance and co-variance for the addition of one observation
        """
        self._covariance_in_window += ((self.window_size + 1.0) / self.window_size *
                                       (xj - self._mean_x_in_window) *
                                       (yj - self._mean_y_in_window))
        self._variance_in_window += ((self.window_size + 1.0) / self.window_size *
                                       (xj - self._mean_x_in_window) ** 2)

    def _remove_observation_to_variances(self, xj, yj):
        """
        Quickly update the variance and co-variance for the deletion of one observation
        """
        self._covariance_in_window -= (self.window_size / (self.window_size - 1.0) *
                                       (xj - self._mean_x_in_window) *
                                       (yj - self._mean_y_in_window))
        self._variance_in_window -= (self.window_size / (self.window_size - 1.0) *
                                       (xj - self._mean_x_in_window) ** 2)

    def _compute_smooth_during_construction(self, xi):
        if self._variance_in_window:
            beta = self._covariance_in_window / self._variance_in_window
            alpha = self._mean_y_in_window - beta * self._mean_x_in_window
            value_of_smooth_here = beta * (xi) + alpha
        else:
            value_of_smooth_here = 0.0
        return value_of_smooth_here

    def _compute_cross_validated_residual_here(self, xi, yi, smooth_here):
        """
        Compute CV residual. 
        
        This is the absolute residual from Eq. 9. 
        """
        residual = abs((yi - smooth_here) / (1.0 - 1.0 / self.window_size -
                                         (xi - self._mean_x_in_window) ** 2 /
                                         self._variance_in_window))
        return residual

class BasicFixedSpanSmootherSlowUpdate(BasicFixedSpanSmoother):
    """
    Uses slow means and variances at each step. Used to validate fast updates
    """

    def _advance_window(self):
        self._window_bound_lower += 1
        self._update_values_in_window()
        self._update_mean_in_window()
        self._update_variance_in_window()


DEFAULT_BASIC_SMOOTHER = BasicFixedSpanSmoother

def perform_smooth(x_values, y_values, span=None, smoother_cls=None):
    """
    Convenience function to run the basic smoother
    """
    if smoother_cls is None:
        smoother_cls = DEFAULT_BASIC_SMOOTHER
    smoother = smoother_cls()
    smoother.specify_data_set(x_values, y_values)
    smoother.set_span(span)
    smoother.compute()
    return smoother


if __name__ == '__main__':
    pass
