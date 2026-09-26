"""
Scatterplot smoother with a fixed span.

Takes x,y scattered data and returns a set of
(x,s) points that form a smoother curve fitting the data with moving least squares estimates.
Similar to a moving average, but with better characteristics. The fundamental issue
with this smoother is that the choice of span (window size) is not known in advance.
The SuperSmoother uses these smoothers to figure out which span is optimal.

This is a Python port of J. Friedman's 1982 fixed-span Smoother [Friedman82]_

Example::

    s = Smoother()
    s.specify_data_set(x, y, sort_data = True)
    s.set_span(0.05)
    s.compute()
    smoothed_y = s.smooth_result

"""

import numpy

try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None

TWEETER_SPAN = 0.05
MID_SPAN = 0.2
BASS_SPAN = 0.5

DEFAULT_SPANS = (TWEETER_SPAN, MID_SPAN, BASS_SPAN)

# Used to numerically stabilize slope calculations for running linear fits (eps in supsmu.f)
VARIANCE_EPS = 1.0e-3


class Smoother:
    """Smoother that accepts data and produces smoother curves that fit the data."""

    def __init__(self):
        """Smoother constructor."""
        self.x = []
        self.y = []

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
        self._neighbors_on_each_side = None
        self._variance_threshold = 0.0

    def add_data_point_xy(self, x, y):
        """Add a new data point to the data set to be smoothed."""
        self.x.append(x)
        self.y.append(y)

    def specify_data_set(self, x_input, y_input, sort_data=False):
        """
        Fully define data by lists of x values and y values.

        This will sort them by increasing x but remember how to unsort them for providing results.

        Parameters
        ----------
        x_input : iterable
            list of floats that represent x
        y_input : iterable
            list of floats that represent y(x) for each x
        sort_data : bool, optional
            If true, the data will be sorted by increasing x values.

        """
        if sort_data:
            # stable sort so tied x-values keep a well-defined order
            self._original_index_of_xvalue = [int(i) for i in numpy.argsort(x_input, kind="stable")]
            x = [x_input[i] for i in self._original_index_of_xvalue]
            y = [y_input[i] for i in self._original_index_of_xvalue]
        else:
            x, y = x_input, y_input

        self.x = x
        self.y = y

    def set_span(self, span):
        """
        Set the window-size for computing the least squares fit.

        Parameters
        ----------
        span : float
            Fraction on data length N to be considered in smoothing

        """
        self._span = span

    def compute(self):
        """Perform the smoothing operations."""
        raise NotImplementedError

    def plot(self, fname=None):
        """
        Plot the input data and resulting smooth.

        Parameters
        ----------
        fname : str, optional
            name of file to produce. If none, will show interactively.

        """
        if not plt:
            raise ImportError("Cannot plot without the matplotlib package")
        plt.figure()
        xy = sorted(zip(self.x, self.smooth_result))
        x, y = zip(*xy)
        plt.plot(x, y, "-")
        plt.plot(self.x, self.y, ".")
        if fname:
            plt.savefig(fname)
        else:
            plt.show()
        plt.close()

    def _store_unsorted_results(self, smooth, residual):
        """Convert sorted smooth/residual back to as-input order."""
        if self._original_index_of_xvalue:
            # data was sorted. Unsort it here.
            self.smooth_result = numpy.zeros(len(self.y))
            self.cross_validated_residual = numpy.zeros(len(residual))
            original_x = numpy.zeros(len(self.y))
            for i, (xval, smooth_val, residual_val) in enumerate(zip(self.x, smooth, residual)):
                original_index = self._original_index_of_xvalue[i]
                original_x[original_index] = xval
                self.smooth_result[original_index] = smooth_val
                self.cross_validated_residual[original_index] = residual_val
            self.x = original_x
        else:
            # no sorting was done. just apply results
            self.smooth_result = smooth
            self.cross_validated_residual = residual


class BasicFixedSpanSmoother(Smoother):
    """
    A basic fixed-span smoother.

    Simple least-squares linear local smoother.

    Uses fast updates of means, variances.
    """

    def compute(self):
        """Perform the smoothing operations."""
        self._compute_window_size()
        self._compute_variance_threshold()
        smooth = []
        residual = []

        x, y = self.x, self.y

        # step through x and y data with a window window_size wide.
        self._update_values_in_window()
        self._update_mean_in_window()
        self._update_variance_in_window()
        for i, (xi, yi) in enumerate(zip(x, y)):
            if (i - self._neighbors_on_each_side) > 0.0 and (
                i + self._neighbors_on_each_side
            ) < len(x):
                self._advance_window()
            smooth_here = self._compute_smooth_during_construction(xi)
            residual_here = self._compute_cross_validated_residual_here(xi, yi, smooth_here)
            if residual_here is None:
                # leverage too high for a meaningful residual. Reuse the previous one like supsmu.f
                residual_here = residual[-1] if residual else 0.0
            smooth.append(smooth_here)
            residual.append(residual_here)

        smooth = self._average_smooth_over_ties(smooth)
        self._store_unsorted_results(smooth, residual)

    def _compute_window_size(self):
        """
        Determine characteristics of symmetric neighborhood with J/2 values on each side.

        Rounds and enforces a minimum of 2 neighbors on each side, like supsmu.f.
        """
        self._neighbors_on_each_side = max(int(0.5 * self._span * len(self.x) + 0.5), 2)
        self.window_size = min(self._neighbors_on_each_side * 2 + 1, len(self.x))

    def _compute_variance_threshold(self):
        """
        Determine the window variance below which the local slope is taken to be zero.

        This is ``vsmlsq`` in supsmu.f, based on the interquartile spread of x.
        """
        num_x = len(self.x)
        lower = max(num_x // 4, 1)
        upper = 3 * lower
        scale = self.x[upper - 1] - self.x[lower - 1]
        while scale <= 0.0 and (upper < num_x or lower > 1):
            if upper < num_x:
                upper += 1
            if lower > 1:
                lower -= 1
            scale = self.x[upper - 1] - self.x[lower - 1]
        self._variance_threshold = (VARIANCE_EPS * scale) ** 2

    def _average_smooth_over_ties(self, smooth):
        """Give all observations with tied x-values the average of their smooth values."""
        smooth = list(smooth)
        start = 0
        while start < len(smooth):
            end = start
            while end + 1 < len(smooth) and self.x[end + 1] <= self.x[end]:
                end += 1
            if end > start:
                tied_mean = sum(smooth[start : end + 1]) / (end + 1 - start)
                smooth[start : end + 1] = [tied_mean] * (end + 1 - start)
            start = end + 1
        return smooth

    def _update_values_in_window(self):
        """Update which values are in the current window."""
        window_bound_upper = self._window_bound_lower + self.window_size
        self._x_in_window = self.x[self._window_bound_lower : window_bound_upper]
        self._y_in_window = self.y[self._window_bound_lower : window_bound_upper]

    def _update_mean_in_window(self):
        """
        Compute mean in window the slow way. useful for first step.

        Considers all values in window

        See Also
        --------
        _add_observation_to_means : fast update of mean for single observation addition
        _remove_observation_from_means : fast update of mean for single observation removal

        """
        self._mean_x_in_window = numpy.mean(self._x_in_window)
        self._mean_y_in_window = numpy.mean(self._y_in_window)

    def _update_variance_in_window(self):
        """
        Compute variance and covariance in window using all values in window (slow).

        See Also
        --------
        _add_observation_to_variances : fast update for single observation addition
        _remove_observation_from_variances : fast update for single observation removal

        """
        self._covariance_in_window = sum(
            [
                (xj - self._mean_x_in_window) * (yj - self._mean_y_in_window)
                for xj, yj in zip(self._x_in_window, self._y_in_window)
            ]
        )

        self._variance_in_window = sum(
            [(xj - self._mean_x_in_window) ** 2 for xj in self._x_in_window]
        )

    def _advance_window(self):
        """Update values in current window and the current window means and variances."""
        x_to_remove, y_to_remove = self._x_in_window[0], self._y_in_window[0]

        self._window_bound_lower += 1
        self._update_values_in_window()
        x_to_add, y_to_add = self._x_in_window[-1], self._y_in_window[-1]

        self._remove_observation(x_to_remove, y_to_remove)
        self._add_observation(x_to_add, y_to_add)

    def _remove_observation(self, x_to_remove, y_to_remove):
        """Remove observation from window, updating means/variance efficiently."""
        self._remove_observation_from_variances(x_to_remove, y_to_remove)
        self._remove_observation_from_means(x_to_remove, y_to_remove)
        self.window_size -= 1

    def _add_observation(self, x_to_add, y_to_add):
        """Add observation to window, updating means/variance efficiently."""
        self._add_observation_to_means(x_to_add, y_to_add)
        self._add_observation_to_variances(x_to_add, y_to_add)
        self.window_size += 1

    def _add_observation_to_means(self, xj, yj):
        """Update the means without recalculating for the addition of one observation."""
        self._mean_x_in_window = (self.window_size * self._mean_x_in_window + xj) / (
            self.window_size + 1.0
        )
        self._mean_y_in_window = (self.window_size * self._mean_y_in_window + yj) / (
            self.window_size + 1.0
        )

    def _remove_observation_from_means(self, xj, yj):
        """Update the means without recalculating for the deletion of one observation."""
        self._mean_x_in_window = (self.window_size * self._mean_x_in_window - xj) / (
            self.window_size - 1.0
        )
        self._mean_y_in_window = (self.window_size * self._mean_y_in_window - yj) / (
            self.window_size - 1.0
        )

    def _add_observation_to_variances(self, xj, yj):
        """
        Quickly update the variance and co-variance for the addition of one observation.

        See Also
        --------
        _update_variance_in_window : compute variance considering full window

        """
        term1 = (self.window_size + 1.0) / self.window_size * (xj - self._mean_x_in_window)
        self._covariance_in_window += term1 * (yj - self._mean_y_in_window)
        self._variance_in_window += term1 * (xj - self._mean_x_in_window)

    def _remove_observation_from_variances(self, xj, yj):
        """Quickly update the variance and co-variance for the deletion of one observation."""
        term1 = self.window_size / (self.window_size - 1.0) * (xj - self._mean_x_in_window)
        self._covariance_in_window -= term1 * (yj - self._mean_y_in_window)
        self._variance_in_window -= term1 * (xj - self._mean_x_in_window)

    def _compute_smooth_during_construction(self, xi):
        """
        Evaluate value of smooth at x-value xi.

        Parameters
        ----------
        xi : float
            Value of x where smooth value is desired

        Returns
        -------
        smooth_here : float
            Value of smooth s(xi)

        """
        beta = 0.0
        if self._variance_in_window > self._variance_threshold:
            beta = self._covariance_in_window / self._variance_in_window
        return beta * (xi - self._mean_x_in_window) + self._mean_y_in_window

    def _compute_cross_validated_residual_here(self, xi, yi, smooth_here):
        """
        Compute cross validated residual.

        This is the absolute residual from Eq. 9. in [1]. Returns None when the
        leverage of this point is 1 or more (can happen with small data sets).
        """
        leverage = 1.0 / self.window_size
        if self._variance_in_window > self._variance_threshold:
            leverage += (xi - self._mean_x_in_window) ** 2 / self._variance_in_window
        denom = 1.0 - leverage
        if denom <= 0.0:
            return None
        return abs((yi - smooth_here) / denom)


class BasicFixedSpanSmootherSlowUpdate(BasicFixedSpanSmoother):
    """Use slow means and variances at each step. Used to validate fast updates."""

    def _advance_window(self):
        self._window_bound_lower += 1
        self._update_values_in_window()
        self._update_mean_in_window()
        self._update_variance_in_window()


DEFAULT_BASIC_SMOOTHER = BasicFixedSpanSmoother


def perform_smooth(x_values, y_values, span=None, smoother_cls=None):
    """
    Run the basic smoother (convenience function).

    Parameters
    ----------
    x_values : iterable
        List of x value observations
    y_ values : iterable
        list of y value observations
    span : float, optional
        Fraction of data to use as the window
    smoother_cls : Class
        The class of smoother to use to smooth the data

    Returns
    -------
    smoother : object
        The smoother object with results stored on it.

    """
    if smoother_cls is None:
        smoother_cls = DEFAULT_BASIC_SMOOTHER
    smoother = smoother_cls()
    smoother.specify_data_set(x_values, y_values)
    smoother.set_span(span)
    smoother.compute()
    return smoother


if __name__ == "__main__":
    pass
