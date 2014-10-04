'''
A variable-span data smoother.

Based on [1]

    [1] J. Friedman, "A Variable Span Smoother", 1984
        http://www.slac.stanford.edu/cgi-wrap/getdoc/slac-pub-3477.pdf
'''
import numpy
from scipy.interpolate import interp1d

from . import smoother
from .smoother import DEFAULT_SPANS, MID_SPAN, BASS_SPAN

BASS_INDEX = DEFAULT_SPANS.index(BASS_SPAN)

class SuperSmoother(smoother.Smoother):
    '''
    Variable-span smoother
    '''

    def __init__(self):
        super(SuperSmoother, self).__init__()

        self._primary_smooths = []
        self._residual_smooths = []
        self._best_span_at_each_point = []
        self._smoothed_best_spans = []
        self._bass_enhancement = 0.0  # should be between 0 and 10.

    def set_bass_enhancement(self, alpha):
        self._bass_enhancement = alpha

    def compute(self):

        self._compute_primary_smooths()
        self._smooth_the_residuals()
        self._select_best_smooth_at_each_point()
        self._enhance_bass()
        self._smooth_best_span_estimates()
        self._apply_best_spans_to_primaries()
        self._build_interpolator(self._x, self.smooth_result)

    def _compute_primary_smooths(self):

        for span in DEFAULT_SPANS:
            smooth = smoother.perform_smooth(self._x, self._y, span)
            self._primary_smooths.append(smooth)

    def _smooth_the_residuals(self):
        """

        "For stability reasons, it turns out to be a little better to smooth
        |r_{i}(J)| against xi" - [1]
        """
        for primary_smooth in self._primary_smooths:
            smooth = smoother.perform_smooth(
                               self._x,
                               numpy.abs(primary_smooth.cross_validated_residual),
                               MID_SPAN)
            self._residual_smooths.append(smooth.smooth_result)

    def _select_best_smooth_at_each_point(self):
        """
        Solve Eq (10) to find the best span for each observation

        Stores index so we can easily grab the best residual smooth, primary smooth, etc.
        """
        for residuals_i in zip(*self._residual_smooths):
            index_of_best_span = residuals_i.index(min(residuals_i))
            self._best_span_at_each_point.append(DEFAULT_SPANS[index_of_best_span])

    def _enhance_bass(self):
        """
        Update best span choices with bass enhancement as requested by user
        (Eq. 11)
        """
        bass_span = DEFAULT_SPANS[BASS_INDEX]
        enhanced_spans = []
        for xi, best_span_here in enumerate(self._best_span_at_each_point):
            best_smooth_index = DEFAULT_SPANS.index(best_span_here)
            ri = ((self._residual_smooths[best_smooth_index][xi]) /
                  (self._residual_smooths[BASS_INDEX][xi]))
            best_span = DEFAULT_SPANS[best_smooth_index]

            enhanced_spans.append(best_span +
                                        (bass_span -
                                         best_span) * ri ** (10.0 - self._bass_enhancement))
        self._best_span_at_each_point = enhanced_spans

    def _smooth_best_span_estimates(self):
        self._smoothed_best_spans = smoother.perform_smooth(
                               self._x, self._best_span_at_each_point, MID_SPAN)

    def _apply_best_spans_to_primaries(self):
        for xi, best_span in enumerate(self._smoothed_best_spans.smooth_result):
            primary_values = [s.smooth_result[xi] for s in self._primary_smooths]
            best_value = numpy.interp(best_span, DEFAULT_SPANS, primary_values)
            self.smooth_result.append(best_value)

        self.smooth_result = numpy.array(self.smooth_result)

