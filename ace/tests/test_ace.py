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

    def test_error_is_decreasing(self):
        err = self.ace._compute_error()
        self.assertFalse(self.ace._error_is_decreasing(err)[0])

    def test_compute_error(self):
        err = self.ace._compute_error()
        self.assertNotAlmostEqual(err, 0.0)

    def test_update_x_transforms(self):
        err = self.ace._compute_error()
        self.ace._update_x_transforms()
        self.assertTrue(self.ace._error_is_decreasing(err)[0])

    def test_update_y_transform(self):
        err = self.ace._compute_error()
        self.ace._update_x_transforms()
        self.ace._update_y_transform()
        self.assertTrue(self.ace._error_is_decreasing(err)[0])

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
