"""
Compare the smoothers and ACE against a line-by-line port of Friedman's FORTRAN.

Several sample sizes are used because the window size rounding in the fixed-span
smoother only shows up for some of them (e.g. not N=200).
"""

import contextlib
import io
import unittest

import numpy

from ace import ace, smoother, supersmoother
from ace.tests import fortran_reference as ref

# pylint: disable=missing-docstring

SAMPLE_SIZES = (37, 100, 200, 500)
TOLERANCE = 1e-9


def friedman82_data(num_obs, seed):
    rng = numpy.random.RandomState(seed)
    x = numpy.sort(rng.uniform(size=num_obs))
    y = numpy.sin(2 * numpy.pi * (1 - x) ** 2) + x * rng.standard_normal(num_obs)
    return x, y


def tied_data(num_obs, seed):
    rng = numpy.random.RandomState(seed)
    x = numpy.sort(rng.randint(0, 6, num_obs).astype(float))
    y = x ** 2 + rng.standard_normal(num_obs)
    return x, y


class TestSmootherAgainstFortran(unittest.TestCase):

    def assert_close(self, actual, expected):
        self.assertLess(numpy.max(numpy.abs(numpy.asarray(actual) - expected)), TOLERANCE)

    def check_fixed_span(self, x, y):
        vsmlsq = ref.variance_threshold(x)
        for span in smoother.DEFAULT_SPANS:
            mine = smoother.perform_smooth(x, y, span)
            expected_smooth, expected_resid = ref.smooth(x, y, span, 1, vsmlsq)
            self.assert_close(mine.smooth_result, expected_smooth)
            self.assert_close(mine.cross_validated_residual, expected_resid)

    def test_fixed_span(self):
        for num_obs in SAMPLE_SIZES:
            self.check_fixed_span(*friedman82_data(num_obs, num_obs))

    def test_fixed_span_with_ties(self):
        self.check_fixed_span(*tied_data(200, 1))

    def test_supersmoother(self):
        for num_obs in SAMPLE_SIZES:
            x, y = friedman82_data(num_obs, num_obs)
            mine = smoother.perform_smooth(x, y, smoother_cls=supersmoother.SuperSmoother)
            self.assert_close(mine.smooth_result, ref.supsmu(x, y))

    def test_supersmoother_with_ties(self):
        x, y = tied_data(200, 2)
        mine = smoother.perform_smooth(x, y, smoother_cls=supersmoother.SuperSmoother)
        self.assert_close(mine.smooth_result, ref.supsmu(x, y))

    def test_supersmoother_constant_x(self):
        x = numpy.ones(50)
        y = numpy.arange(50.0)
        mine = smoother.perform_smooth(x, y, smoother_cls=supersmoother.SuperSmoother)
        self.assert_close(mine.smooth_result, ref.supsmu(x, y))

    def test_supersmoother_bass(self):
        x, y = friedman82_data(200, 3)
        for alpha in (3.0, 8.0):
            mine = supersmoother.SuperSmoother()
            mine.set_bass_enhancement(alpha)
            mine.specify_data_set(x, y)
            mine.compute()
            self.assert_close(mine.smooth_result, ref.supsmu(x, y, alpha=alpha))


class TestAceAgainstFortran(unittest.TestCase):

    def check_ace(self, x_values, y_values):
        solver = ace.ACESolver()
        solver.specify_data_set(x_values, y_values)
        with contextlib.redirect_stdout(io.StringIO()):
            solver.solve()
        tx_expected, ty_expected, rsq_expected = ref.mace(x_values, y_values)
        self.assertAlmostEqual(solver.rsq, rsq_expected, places=9)
        self.assertLess(numpy.max(numpy.abs(solver.y_transform - ty_expected)), TOLERANCE)
        for i, x_transform in enumerate(solver.x_transforms):
            self.assertLess(numpy.max(numpy.abs(x_transform - tx_expected[:, i])), TOLERANCE)

    def test_breiman85_example1(self):
        for num_obs in (100, 200):
            rng = numpy.random.RandomState(num_obs)
            x = numpy.cbrt(rng.standard_normal(num_obs))
            y = numpy.exp(x ** 3 + rng.standard_normal(num_obs))
            self.check_ace([x], y)

    def test_breiman85_example3(self):
        for num_obs in (100, 200):
            rng = numpy.random.RandomState(num_obs)
            x1 = rng.uniform(-1, 1, num_obs)
            x2 = rng.uniform(-1, 1, num_obs)
            self.check_ace([x1, x2], x1 * x2)


if __name__ == '__main__':
    unittest.main()
