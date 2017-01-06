"""Unit tests for ace model."""

import unittest
import os

from ace import model
from ace.samples import breiman85
from ace.samples import wang04

# pylint: disable=protected-access, missing-docstring

class TestModel(unittest.TestCase):

    def setUp(self):
        self.model = model.Model()

    def tearDown(self):
        pass

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
        fname = os.path.join(os.path.dirname(__file__), 'sample_xy_input.txt')
        self.model.ace.write_input_to_file(fname)

        model2 = model.Model()
        model2.build_model_from_txt(fname)

        val = self.model.eval([0.5])
        val2 = model2.eval([0.5])
        self.assertAlmostEqual(val, val2, 2)

        model2.ace.write_transforms_to_file()

    def test_smaller_dataset(self):
        x, y = wang04.build_sample_ace_problem_wang04(N=50)
        self.model.build_model_from_xy(x, y)
        val = self.model.eval([0.5, 0.3, 0.2, 0.1, 0.0])
        self.assertGreater(val, 0.0)


if __name__ == "__main__":
    unittest.main()
