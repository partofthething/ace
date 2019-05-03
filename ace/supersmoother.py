"""
A variable-span data smoother.

This uses the fixed-span smoother to determine
a changing optimal span for the data based on cross-validated residuals. It
is an adaptive smoother that requires several passes over the data.

The SuperSmoother provides a mechanism to evaluate the conditional expectations
in the ACE algorithm.

Based on [Friedman82]_.

Example::

    s = SuperSmoother()
    s.specify_data_set(x, y, sort_data = True)
    s.compute()
    smoothed_y = s.smooth_result
"""

import numpy
from matplotlib import pyplot as plt

from . import smoother
from .smoother import DEFAULT_SPANS, MID_SPAN, BASS_SPAN, TWEETER_SPAN


BASS_INDEX = DEFAULT_SPANS.index(BASS_SPAN)


class SuperSmoother(smoother.Smoother):
    """Variable-span smoother."""

    def __init__(self):
        """Construct a SuperSmoother."""
        super(SuperSmoother, self).__init__()

        self._primary_smooths = []
        self._residual_smooths = []
        self._best_span_at_each_point = []
        self._smoothed_best_spans = None
        self._bass_enhancement = 0.0  # should be between 0 and 10.

    def set_bass_enhancement(self, alpha):
        """
        Bass enhancement amplifies the bass span.

        This gives the resulting smooth a smoother look, which is sometimes desirable if
        the underlying mechanisms are known to be smooth.
        """
        self._bass_enhancement = alpha

    def compute(self):
        """Run the SuperSmoother."""
        self._compute_primary_smooths()
        self._smooth_the_residuals()
        self._select_best_smooth_at_each_point()
        self._enhance_bass()
        self._smooth_best_span_estimates()
        self._apply_best_spans_to_primaries()
        self._smooth_interpolated_smooth()
        self._store_unsorted_results(self.smooth_result, numpy.zeros(len(self.smooth_result)))

    def _compute_primary_smooths(self):
        """Compute fixed-span smooths with all of the default spans."""
        for span in DEFAULT_SPANS:
            smooth = smoother.perform_smooth(self.x, self.y, span)
            self._primary_smooths.append(smooth)

    def _smooth_the_residuals(self):
        """
        Apply the MID_SPAN to the residuals of the primary smooths.

        "For stability reasons, it turns out to be a little better to smooth
        |r_{i}(J)| against xi" - [1]
        """
        for primary_smooth in self._primary_smooths:
            smooth = smoother.perform_smooth(self.x,
                                             primary_smooth.cross_validated_residual,
                                             MID_SPAN)
            self._residual_smooths.append(smooth.smooth_result)

    def _select_best_smooth_at_each_point(self):
        """
        Solve Eq (10) to find the best span for each observation.

        Stores index so we can easily grab the best residual smooth, primary smooth, etc.
        """
        for residuals_i in zip(*self._residual_smooths):
            index_of_best_span = residuals_i.index(min(residuals_i))
            self._best_span_at_each_point.append(DEFAULT_SPANS[index_of_best_span])

    def _enhance_bass(self):
        """Update best span choices with bass enhancement as requested by user (Eq. 11)."""
        if not self._bass_enhancement:
            # like in supsmu, skip if alpha=0
            return
        bass_span = DEFAULT_SPANS[BASS_INDEX]
        enhanced_spans = []
        for i, best_span_here in enumerate(self._best_span_at_each_point):
            best_smooth_index = DEFAULT_SPANS.index(best_span_here)
            best_span = DEFAULT_SPANS[best_smooth_index]
            best_span_residual = self._residual_smooths[best_smooth_index][i]
            bass_span_residual = self._residual_smooths[BASS_INDEX][i]
            if 0 < best_span_residual < bass_span_residual:
                ri = best_span_residual / bass_span_residual
                bass_factor = ri ** (10.0 - self._bass_enhancement)
                enhanced_spans.append(best_span + (bass_span - best_span) * bass_factor)
            else:
                enhanced_spans.append(best_span)
        self._best_span_at_each_point = enhanced_spans

    def _smooth_best_span_estimates(self):
        """Apply a MID_SPAN smooth to the best span estimates at each observation."""
        self._smoothed_best_spans = smoother.perform_smooth(self.x,
                                                            self._best_span_at_each_point,
                                                            MID_SPAN)

    def _apply_best_spans_to_primaries(self):
        """
        Apply best spans.

        Given the best span, interpolate to compute the best smoothed value
        at each observation.
        """
        self.smooth_result = []
        for xi, best_span in enumerate(self._smoothed_best_spans.smooth_result):
            primary_values = [s.smooth_result[xi] for s in self._primary_smooths]
            # pylint: disable=no-member
            best_value = numpy.interp(best_span, DEFAULT_SPANS, primary_values)
            self.smooth_result.append(best_value)

    def _smooth_interpolated_smooth(self):
        """
        Smooth interpolated results with tweeter span.

        A final step of the supersmoother is to smooth the interpolated values with
        the tweeter span. This is done in Breiman's supsmu.f but is not explicitly
        discussed in the publication. This step is necessary to match
        the FORTRAN version perfectly.
        """
        smoothed_results = smoother.perform_smooth(self.x,
                                                   self.smooth_result,
                                                   TWEETER_SPAN)
        self.smooth_result = smoothed_results.smooth_result

class SuperSmootherWithPlots(SuperSmoother):
    """Auxiliary subclass for researching/understanding the SuperSmoother."""

    def _compute_primary_smooths(self):
        super(SuperSmootherWithPlots, self)._compute_primary_smooths()
        plt.figure()
        for smooth in self._primary_smooths:
            plt.plot(self.x, smooth.smooth_result)
        plt.plot(self.x, self.y, '.')
        plt.savefig('primary_smooths.png')
        plt.close()

    def _smooth_the_residuals(self):
        super(SuperSmootherWithPlots, self)._smooth_the_residuals()
        plt.figure()
        for residual, span in zip(self._residual_smooths, smoother.DEFAULT_SPANS):
            plt.plot(self.x, residual, label='{0}'.format(span))
        plt.legend(loc='upper left')
        plt.savefig('residual_smooths.png')
        plt.close()

    def _select_best_smooth_at_each_point(self):
        super(SuperSmootherWithPlots, self)._select_best_smooth_at_each_point()
        plt.figure()
        plt.plot(self.x, self._best_span_at_each_point, label='Fresh')

    def _enhance_bass(self):
        super(SuperSmootherWithPlots, self)._enhance_bass()
        plt.plot(self.x, self._best_span_at_each_point, label='Enhanced bass')

    def _smooth_best_span_estimates(self):
        super(SuperSmootherWithPlots, self)._smooth_best_span_estimates()
        plt.plot(self.x, self._smoothed_best_spans.smooth_result, label='Smoothed')
        plt.legend(loc='upper left')
        plt.savefig('best_spans.png')
        plt.close()
