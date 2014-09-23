'''
Python port of J. Friedman's 1984 smoother

[1] J. Friedman, "A Variable Span Smoother", 1984 
        http://www.slac.stanford.edu/cgi-wrap/getdoc/slac-pub-3477.pdf
'''

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
        self.__covariance_in_window = 0.0
        self.__variance_in_window = 0.0

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

    def specify_data_set(self, xvalues, yvalues):
        """
        Fully define data by lists of x values and y values

        Parameters
        ----------
        xValues : iterable
            list of floats that represent x
        yValues : iterable
            list of floats that represent y(x) for each x
        """
        self._x = xvalues
        self._y = yvalues

    def set_span(self, span):
        self._span = span

    def get_dataset_sorted(self):
        xy = zip(self._x, self._y)
        xy.sort()
        x, y = zip(*xy)  # pylint: disable=star-args
        self._original_index_of_xvalue = [list(self._x).index(xi) for xi in x]
        return numpy.array(x), numpy.array(y)

    def compute(self):
        raise NotImplementedError

    def plot(self):
        pylab.figure()
        pylab.plot(self._x, self.smooth_result)
        pylab.show()

class BasicFixedSpanSmoother(Smoother):
    """
    A basic fixed-span smoother

    Simple least-squares linear local smoother. 
    
    Notes
    -----
    Can be easily upgraded so that means can be updated rather than recalculated, improving
    performance, if needed. 
    
    """
    def compute(self):
        """
        
        """
        self._compute_window_size()
        smooth = []
        residual = []

        x, y = self.get_dataset_sorted()

        # step through x and y data with a window window_size wide.
        window_bound_lower = 0
        x_values_in_window, y_values_in_window = self._get_values_in_window(x, y, window_bound_lower)
        self._update_mean_in_window(x_values_in_window, y_values_in_window)
        for i, (xi, yi) in enumerate(zip(x, y)):


            self._update_variance_in_window(x_values_in_window, y_values_in_window)

            smooth_here = self._compute_smooth_here(xi, yi)
            residual_here = self._compute_cross_validated_residual_here(xi, yi, smooth_here)
            smooth.append(smooth_here)
            residual.append(residual_here)

            if i - self.window_size / 2.0 > 0.0 and i + self.window_size / 2.0 <= len(x):
                window_bound_lower += 1
                self._remove_observation_from_means(x_values_in_window[0],
                                                    y_values_in_window[0])
                x_values_in_window, y_values_in_window = self._get_values_in_window(x, y, window_bound_lower)
                self._add_observation_to_means(x_values_in_window[-1],
                                               y_values_in_window[-1])

        self.smooth_result = numpy.zeros(len(self._y))
        for i, smoothVal in enumerate(smooth):
            self.smooth_result[self._original_index_of_xvalue[i]] = smoothVal
        self.cross_validated_residual = numpy.array(residual)

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

    def _compute_smooth_here(self, xi, yi):
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

def perform_smooth(x_values, y_values, span=0.0, smoother_class=BasicFixedSpanSmoother):
    """
    Convenience function to run the basic smoother
    """
    smoother = smoother_class()
    smoother.specify_data_set(x_values, y_values)
    smoother.set_span(span)
    smoother.compute()
    return smoother


if __name__ == '__main__':
    pass
