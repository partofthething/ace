"""
The Model module is a front-end to the :py:mod:`ace.ace` module.

ACE itself just gives transformations back as discontinuous data points.
This module loads data, runs ACE, and then performs interpolations on the results,
giving the user continuous functions that may be evaluated at any point
within the range trained.

This is a convenience/frontend/demo module. If you want to control ACE yourself,
you may want to just use the ace module manually.

"""

from scipy.interpolate import interp1d

from . import ace


def read_column_data_from_txt(fname):
    """
    Read data from a simple text file.

    Format should be just numbers.
    First column is the dependent variable. others are independent.
    Whitespace delimited.

    Returns
    -------
    x_values : list
        List of x columns
    y_values : list
        list of y values

    """
    datafile = open(fname)
    datarows = []
    for line in datafile:
        datarows.append([float(li) for li in line.split()])
    datacols = list(zip(*datarows))
    x_values = datacols[1:]
    y_values = datacols[0]

    return x_values, y_values


class Model(object):
    """A continuous model of data based on ACE regressions."""

    def __init__(self):
        """Model constructor."""
        self.ace = ace.ACESolver()
        self.phi_continuous = None
        self.inverse_theta_continuous = None

    def build_model_from_txt(self, fname):
        """
        Construct the model and perform regressions based on data in a txt file.

        Parameters
        ----------
        fname : str
            The name of the file to load.

        """
        x_values, y_values = read_column_data_from_txt(fname)
        self.build_model_from_xy(x_values, y_values)

    def build_model_from_xy(self, x_values, y_values):
        """Construct the model and perform regressions based on x, y data."""
        self.init_ace(x_values, y_values)
        self.run_ace()
        self.build_interpolators()

    def init_ace(self, x_values, y_values):
        """Specify data for the ACE solver object."""
        self.ace.specify_data_set(x_values, y_values)

    def run_ace(self):
        """Perform the ACE calculation."""
        self.ace.solve()

    def build_interpolators(self):
        """Compute 1-D interpolation functions for all the transforms so they're continuous.."""
        self.phi_continuous = []
        for xi, phii in zip(self.ace.x, self.ace.x_transforms):
            self.phi_continuous.append(
                interp1d(
                    x=xi,
                    y=phii,
                    bounds_error=False,
                    fill_value=(min(phii), max(phii))))
        self.inverse_theta_continuous = interp1d(
            x=self.ace.y_transform,
            y=self.ace.y,
            bounds_error=False,
            fill_value=(min(self.ace.y), max(self.ace.y)))

    def eval(self, x_values):
        """
        Evaluate the ACE regression at any combination of independent variable values.

        Parameters
        ----------
        x_values : iterable
            a float x-value for each independent variable, e.g. (1.5, 2.5)

        """
        if len(x_values) != len(self.phi_continuous):
            raise ValueError(
                'x_values must have length equal to the number of independent variables '
                '({0}) rather than {1}.'.format(
                    len(self.phi_continuous), len(x_values)))

        sum_phi = sum(
            [phi(xi) for phi, xi in zip(self.phi_continuous, x_values)])
        return self.inverse_theta_continuous(sum_phi)


if __name__ == '__main__':
    pass
