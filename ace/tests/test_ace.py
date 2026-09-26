"""
Unit tests for ACE methods.

These implicitly cover the SuperSmoother as well, but they don't validate it.
"""

import unittest

import ace.ace
import ace.samples.breiman85

# pylint: disable=protected-access, missing-docstring

class TestAce(unittest.TestCase):
    """Tests."""

    def setUp(self):
        self.ace = ace.ace.ACESolver()
        x, y = ace.samples.breiman85.build_sample_ace_problem_breiman85()
        self.ace.specify_data_set(x, y)
        self.ace._initialize()

    def test_compute_sorted_indices(self):
        yprevious = self.ace.y[self.ace._yi_sorted[0]]
        for yi in self.ace._yi_sorted[1:]:
            yhere = self.ace.y[yi]
            self.assertGreater(yhere, yprevious)
            yprevious = yhere
        xprevious = self.ace.x[0][self.ace._xi_sorted[0][0]]
        for xi in self.ace._xi_sorted[1:]:
            xhere = self.ace.x[xi]
            self.assertGreater(xhere, xprevious)
            xprevious = xhere

    def test_compute_error(self):
        err = self.ace._compute_error()
        self.assertNotAlmostEqual(err, 0.0)

    def test_initial_x_transforms_are_scaled_linear_fit(self):
        """Initial phi should be the least-squares linear fit of theta on x (like mace.f)."""
        x_centered = self.ace.x[0] - self.ace.x[0].mean()
        coeff = x_centered.dot(self.ace.y_transform) / x_centered.dot(x_centered)
        self.assertLess(max(abs(self.ace.x_transforms[0] - coeff * x_centered)), 1e-6)

    def test_update_x_transforms(self):
        err = self.ace._compute_error()
        self.ace._update_x_transforms()
        self.assertLess(self.ace._compute_error(), err)

    def test_update_x_transforms_rejects_worse(self):
        """A new phi that doesn't improve R^2 should be rejected."""
        self.ace.rsq = 1.0
        before = [xt.copy() for xt in self.ace.x_transforms]
        self.ace._update_x_transforms()
        for xt_before, xt_after in zip(before, self.ace.x_transforms):
            self.assertTrue((xt_before == xt_after).all())

    def test_update_y_transform(self):
        self.ace._update_x_transforms()
        err = self.ace._compute_error()
        self.ace._update_y_transform()
        self.assertLess(self.ace._compute_error(), err)

    def test_solve_respects_maxit(self):
        self.ace.maxit = 2
        self.ace.delrsq = -1.0  # never converge
        self.ace.solve()
        self.assertEqual(self.ace._outer_iters, 2)

    def test_sort_vector(self):
        data = [5, 1, 4, 6]
        increasing = [1, 2, 0, 3]
        dsort = ace.ace.sort_vector(data, increasing)
        for item1, item2 in zip(sorted(data), dsort):
            self.assertEqual(item1, item2)

    def test_unsort_vector(self):
        unsorted = [5, 1, 4, 6]
        data = [1, 4, 5, 6]
        increasing = [1, 2, 0, 3]
        dunsort = ace.ace.unsort_vector(data, increasing)
        for item1, item2 in zip(dunsort, unsorted):
            self.assertEqual(item1, item2)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
