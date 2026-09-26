"""Unit tests for ace model."""

import tempfile
import unittest
from pathlib import Path

from ace import model
from ace.samples import breiman85, wang04


class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = model.Model()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_build_model_from_xy(self):
        x, y = breiman85.build_sample_ace_problem_breiman85()
        self.model.build_model_from_xy(x, y)

    def test_eval_1d(self):
        x, y = breiman85.build_sample_ace_problem_breiman85()
        self.model.build_model_from_xy(x, y)
        val = self.model.eval([0.5])
        self.assertGreater(val, 0.0)

    def test_eval_multiple(self):
        x, y = wang04.build_sample_ace_problem_wang04()
        self.model.build_model_from_xy(x, y)
        val = self.model.eval([0.5, 0.3, 0.2, 0.1, 0.0])
        self.assertGreater(val, 0.0)

    def test_read_column_data_from_txt(self):
        x, y = breiman85.build_sample_ace_problem_breiman85()
        self.model.build_model_from_xy(x, y)
        fname = self.tmpdir / "sample_xy_input.txt"
        self.model.ace.write_input_to_file(fname)

        model2 = model.Model()
        model2.build_model_from_txt(fname)

        val = self.model.eval([0.5])
        val2 = model2.eval([0.5])
        self.assertAlmostEqual(val, val2, 2)

        transforms_fname = self.tmpdir / "ace_transforms.txt"
        model2.ace.write_transforms_to_file(transforms_fname)
        self.assertGreater(transforms_fname.stat().st_size, 0)

    def test_linear_interpolator(self):
        """Unsorted input is handled and end values are held beyond the data."""
        interp = model.linear_interpolator([3.0, 1.0, 2.0], [30.0, 10.0, 25.0])
        self.assertAlmostEqual(interp(1.5), 17.5)
        self.assertAlmostEqual(interp(2.5), 27.5)
        self.assertAlmostEqual(interp(0.0), 10.0)
        self.assertAlmostEqual(interp(9.0), 30.0)

    def test_smaller_dataset(self):
        x, y = wang04.build_sample_ace_problem_wang04(N=50)
        self.model.build_model_from_xy(x, y)
        val = self.model.eval([0.5, 0.3, 0.2, 0.1, 0.0])
        self.assertGreater(val, 0.0)


if __name__ == "__main__":
    unittest.main()
