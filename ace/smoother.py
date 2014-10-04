'''
Python port of J. Friedman's 1984 smoother

[1] J. Friedman, "A Variable Span Smoother", 1984 
        http://www.slac.stanford.edu/cgi-wrap/getdoc/slac-pub-3477.pdf
'''

import numpy
import pylab
from scipy.interpolate import interp1d

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

    def plot(self):
        pylab.figure()
        pylab.plot(self._x, self.smooth_result)
        pylab.show()

    def _store_unsorted_results(self, smooth, residual):
        """
        Convert sorted smooth/residual back to as-input order
        """
        self.smooth_result = numpy.zeros(len(self._y))
        self.cross_validated_residual = numpy.zeros(len(residual))
        for i, (smooth_val, residual_val) in enumerate(zip(smooth, residual)):
            original_index = self._original_index_of_xvalue[i]
            self.smooth_result[original_index] = smooth_val
            self.cross_validated_residual[original_index] = residual_val



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
        window_bound_lower = 0
        x_values_in_window, y_values_in_window = self._get_values_in_window(x, y,
                                                                            window_bound_lower)
        self._update_mean_in_window(x_values_in_window, y_values_in_window)
        self._update_variance_in_window(x_values_in_window, y_values_in_window)
        for i, (xi, yi) in enumerate(zip(x, y)):
            smooth_here = self._compute_smooth_during_construction(xi)
            residual_here = self._compute_cross_validated_residual_here(xi, yi, smooth_here)
            smooth.append(smooth_here)
            residual.append(residual_here)
            if i - self.window_size / 2.0 > 0.0 and i + self.window_size / 2.0 <= len(x):
                window_bound_lower, x_values_in_window, y_values_in_window = self._update_window(x, y, window_bound_lower, x_values_in_window, y_values_in_window)
        self._store_unsorted_results(smooth, residual)

    def _compute_window_size(self):
        self.window_size = int(len(self._x) * self._span)  # number of nearest neighbors

    def _get_values_in_window(self, x, y, window_bound_lower):
        window_bound_upper = window_bound_lower + self.window_size
        x_values_in_window = x[window_bound_lower:window_bound_upper]
        y_values_in_window = y[window_bound_lower:window_bound_upper]
        return x_values_in_window, y_values_in_window

    def _update_mean_in_window(self, x_values_in_window, y_values_in_window):
        self._mean_x_in_window = numpy.mean(x_values_in_window)
        self._mean_y_in_window = numpy.mean(y_values_in_window)

    def _update_variance_in_window(self, x_values_in_window, y_values_in_window):
        self._covariance_in_window = sum([(xj - self._mean_x_in_window) *
                                          (yj - self._mean_y_in_window)
                      for xj, yj in zip(x_values_in_window, y_values_in_window)])

        self._variance_in_window = sum([(xj - self._mean_x_in_window) ** 2 for xj
                                        in x_values_in_window])

    def _update_window(self, x, y, window_bound_lower, x_values_in_last_window, y_values_in_last_window):
        """
        Update values in current window and the current window means and variances. 
        """
        x_to_remove, y_to_remove = x_values_in_last_window[0], y_values_in_last_window[0]

        window_bound_lower += 1

        x_values_in_window, y_values_in_window = self._get_values_in_window(x, y, window_bound_lower)
        x_to_add, y_to_add = x_values_in_window[-1], y_values_in_window[-1]

        self._remove_observation(x_to_remove, y_to_remove)
        self._add_observation(x_to_add, y_to_add)

        return window_bound_lower, x_values_in_window, y_values_in_window

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
            m = self._covariance_in_window / self._variance_in_window
            b = self._mean_y_in_window - m * self._mean_x_in_window
            value_of_smooth_here = m * (xi) + b
        else:
            value_of_smooth_here = 0.0

        return value_of_smooth_here

    def _compute_cross_validated_residual_here(self, xi, yi, smooth_here):
        """
        Eq. (9)
        """
        residual = (yi - smooth_here) / (1.0 - 1.0 / self.window_size -
                                         (xi - self._mean_x_in_window) ** 2 /
                                         self._variance_in_window)
        return residual

class BasicFixedSpanSmootherSlowUpdate(BasicFixedSpanSmoother):
    """
    Uses slow means and variances at each step. Used to validate fast updates
    """

    def _update_window(self, x, y, window_bound_lower, x_values_in_last_window, y_values_in_last_window):
        window_bound_lower += 1
        x_values_in_window, y_values_in_window = self._get_values_in_window(x, y, window_bound_lower)
        self._update_mean_in_window(x_values_in_window, y_values_in_window)
        self._update_variance_in_window(x_values_in_window, y_values_in_window)
        return window_bound_lower, x_values_in_window, y_values_in_window


DEFAULT_BASIC_SMOOTHER = BasicFixedSpanSmoother

def perform_smooth(x_values, y_values, span=0.0, smoother_cls=None):
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
