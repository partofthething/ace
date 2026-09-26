r"""
The Alternating Conditional Expectation (ACE) algorithm.

ACE was invented by L. Breiman and J. Friedman [Breiman85]_. It is a powerful
way to perform multidimensional regression without assuming
any functional form of the model. Given a data set:

    :math:`y = f(X)`

where :math:`X` is made up of a number of independent variables xi, ACE
will tell you how :math:`y` varies vs. each of the individual independents :math:`xi`.
This can be used to:

    * Understand the relative shape and magnitude of y's dependence on each xi
    * Produce a lightweight surrogate model of a more complex response
    * other stuff

"""

from pathlib import Path

import numpy

try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None

from .smoother import perform_smooth
from .supersmoother import SuperSmoother

# Iteration controls. These defaults match Friedman's mace.f
DEFAULT_DELRSQ = 0.01
DEFAULT_MAXIT = 20
DEFAULT_NTERM = 3


class ACESolver:
    """
    The Alternating Conditional Expectation algorithm to perform regressions.

    The iteration control follows Friedman's mace.f rather than the simpler
    description in [Breiman85]_.

    Parameters
    ----------
    delrsq : float, optional
        Termination threshold. Iteration stops when R^2 changes by less than this
        over ``nterm`` consecutive outer iterations.
    maxit : int, optional
        Maximum number of inner and of outer iterations.
    nterm : int, optional
        Number of consecutive outer iterations considered for convergence.

    """

    def __init__(self, delrsq=DEFAULT_DELRSQ, maxit=DEFAULT_MAXIT, nterm=DEFAULT_NTERM):
        """Solver constructor."""
        self.delrsq = delrsq
        self.maxit = maxit
        self.nterm = nterm
        self.rsq = 0.0
        self.x = []
        self.y = None
        self._xi_sorted = None
        self._yi_sorted = None
        self.x_transforms = None
        self.y_transform = None
        self._smoother_cls = SuperSmoother
        self._outer_iters = 0
        self._inner_iters = 0

    def specify_data_set(self, x_input, y_input):
        """
        Define input to ACE.

        Parameters
        ----------
        x_input : list
            list of iterables, one for each independent variable
        y_input : array
            the dependent observations

        """
        self.x = x_input
        self.y = y_input

    def solve(self):
        """Run the ACE calculational loop."""
        self._initialize()
        # like mace.f, seed the history with huge values so at least nterm iterations run
        rsq_history = [100.0] * self.nterm
        self._outer_iters = 0
        while True:
            print(
                f"* Starting outer iteration {self._outer_iters:03d}. "
                f"Current R^2 = {self.rsq:12.5E}"
            )
            self._iterate_to_update_x_transforms()
            self._update_y_transform()
            self.rsq = 1.0 - self._compute_error()
            rsq_history[self._outer_iters % self.nterm] = self.rsq
            self._outer_iters += 1
            if (
                max(rsq_history) - min(rsq_history) <= self.delrsq
                or self._outer_iters >= self.maxit
            ):
                break

    def _initialize(self):
        """
        Set up and normalize initial data once input data is specified.

        Like mace.f, theta starts as standardized y and each phi starts as its
        centered x, linearly scaled to best fit theta.
        """
        self.y_transform = numpy.array(self.y, dtype=float)
        self.y_transform -= numpy.mean(self.y_transform)
        self.y_transform /= numpy.std(self.y_transform)
        self.x_transforms = [numpy.array(xi, dtype=float) - numpy.mean(xi) for xi in self.x]
        self._scale_x_transforms()
        self.rsq = 0.0
        self._compute_sorted_indices()

    def _scale_x_transforms(self):
        """
        Scale the initial x transforms by a linear least-squares fit to theta.

        Port of the ``scale`` subroutine in mace.f, which uses a few conjugate
        gradient passes to find coefficients c minimizing E[(theta - sum c_i phi_i)^2].
        """
        phis = numpy.array(self.x_transforms).T
        num_obs, num_vars = phis.shape
        coeffs = numpy.zeros(num_vars)
        last_direction = numpy.zeros(num_vars)
        last_gradient_sq = 1.0
        for _pass in range(num_vars):
            previous_coeffs = coeffs.copy()
            for step in range(num_vars):
                residual = self.y_transform - phis.dot(coeffs)
                gradient = -2.0 * residual.dot(phis) / num_obs
                gradient_sq = gradient.dot(gradient)
                if gradient_sq <= 0.0:
                    break
                if step == 0:
                    direction = -gradient
                else:
                    direction = -gradient + gradient_sq / last_gradient_sq * last_direction
                last_gradient_sq = gradient_sq
                projected = phis.dot(direction)
                coeffs += projected.dot(residual) / projected.dot(projected) * direction
                last_direction = direction
            if numpy.max(numpy.abs(coeffs - previous_coeffs)) < self.delrsq:
                break
        self.x_transforms = [
            coeff * phi for coeff, phi in zip(coeffs, self.x_transforms, strict=True)
        ]

    def _compute_sorted_indices(self):
        """
        Sort data from the perspective of each column.

        if self._x[0][3] is the 9th-smallest value in self._x[0], then  _xi_sorted[3] = 8

        We only have to sort the data once.
        """
        sorted_indices = []
        for to_sort in [self.y, *self.x]:
            data_w_indices = [(val, i) for (i, val) in enumerate(to_sort)]
            data_w_indices.sort()
            sorted_indices.append([i for val, i in data_w_indices])
        # save in meaningful variable names
        self._yi_sorted = sorted_indices[0]  # list (like self.y)
        self._xi_sorted = sorted_indices[1:]  # list of lists (like self.x)

    def _compute_error(self):
        """Compute unexplained error."""
        sum_x = sum(self.x_transforms)
        return sum((self.y_transform - sum_x) ** 2) / len(sum_x)

    def _iterate_to_update_x_transforms(self):
        """
        Perform the inner iteration.

        Stops when R^2 improves by no more than delrsq, after maxit passes, or after
        a single pass when there's only one independent variable (like mace.f).
        """
        self._inner_iters = 0
        while True:
            print(
                f"  Starting inner iteration {self._inner_iters:03d}. "
                f"Current R^2 = {self.rsq:12.5E}"
            )
            rsq_before = self.rsq
            self._update_x_transforms()
            self._inner_iters += 1
            if (
                len(self.x) == 1
                or self.rsq - rsq_before <= self.delrsq
                or self._inner_iters >= self.maxit
            ):
                break

    def _update_x_transforms(self):
        """
        Compute a new set of x-transform functions phik.

        phik(xk) = theta(y) - sum of phii(xi) over i!=k

        This is the first of the eponymous conditional expectations. The conditional
        expectations are computed using the SuperSmoother.

        Like mace.f, each new phik is only accepted if it improves R^2.
        """
        # start by subtracting all transforms
        theta_minus_phis = self.y_transform - numpy.sum(self.x_transforms, axis=0)

        # add one transform at a time so as to exclude it from the subtracted sum
        for xtransform_index in range(len(self.x_transforms)):
            xtransform = self.x_transforms[xtransform_index]
            sorted_data_indices = self._xi_sorted[xtransform_index]
            xk_sorted = sort_vector(self.x[xtransform_index], sorted_data_indices)
            xtransform_sorted = sort_vector(xtransform, sorted_data_indices)
            theta_minus_phis_sorted = sort_vector(theta_minus_phis, sorted_data_indices)

            # minimize sums by just adding in the phik where i!=k here.
            to_smooth = theta_minus_phis_sorted + xtransform_sorted

            smoother = perform_smooth(xk_sorted, to_smooth, smoother_cls=self._smoother_cls)
            updated_x_transform_smooth = numpy.array(smoother.smooth_result)
            updated_x_transform_smooth -= numpy.mean(updated_x_transform_smooth)

            rsq = 1.0 - numpy.mean((to_smooth - updated_x_transform_smooth) ** 2)
            if rsq <= self.rsq:
                continue
            self.rsq = rsq

            # store updated transform in the order of the original data
            unsorted_xt = unsort_vector(updated_x_transform_smooth, sorted_data_indices)
            self.x_transforms[xtransform_index] = unsorted_xt

            # update main expession with new smooth. This was done in the original FORTRAN
            tmp_unsorted = unsort_vector(to_smooth, sorted_data_indices)
            theta_minus_phis = tmp_unsorted - unsorted_xt

    def _update_y_transform(self):
        """
        Update the y-transform (theta).

        y-transform theta is forced to have mean = 0 and stddev = 1.

        This is the second conditional expectation
        """
        # sort all phis wrt increasing y.
        sorted_data_indices = self._yi_sorted
        sorted_xtransforms = []
        for xt in self.x_transforms:
            sorted_xt = sort_vector(xt, sorted_data_indices)
            sorted_xtransforms.append(sorted_xt)

        sum_of_x_transformations_choppy = numpy.sum(sorted_xtransforms, axis=0)
        y_sorted = sort_vector(self.y, sorted_data_indices)
        smooth = perform_smooth(
            y_sorted, sum_of_x_transformations_choppy, smoother_cls=self._smoother_cls
        )
        sum_of_x_transformations_smooth = smooth.smooth_result

        sum_of_x_transformations_smooth -= numpy.mean(sum_of_x_transformations_smooth)
        sum_of_x_transformations_smooth /= numpy.std(sum_of_x_transformations_smooth)

        # unsort to save in the original data
        self.y_transform = unsort_vector(sum_of_x_transformations_smooth, sorted_data_indices)

    def write_input_to_file(self, fname="ace_input.txt"):
        """Write y and x values used in this run to a space-delimited txt file."""
        self._write_columns(fname, self.x, self.y)

    def write_transforms_to_file(self, fname="ace_transforms.txt"):
        """Write y and x transforms used in this run to a space-delimited txt file."""
        self._write_columns(fname, self.x_transforms, self.y_transform)

    def _write_columns(self, fname, xvals, yvals):
        with Path(fname).open("w") as output_file:
            alldata = [yvals, *xvals]
            for datai in zip(*alldata, strict=True):
                yline = f"{datai[0]: 15.9E} "
                xline = " ".join([f"{xii: 15.9E}" for xii in datai[1:]])
                output_file.write("".join([yline, xline, "\n"]))


def sort_vector(data, indices_of_increasing):
    """Permutate 1-d data using given indices."""
    return numpy.asarray(data)[indices_of_increasing]


def unsort_vector(data, indices_of_increasing):
    """Upermutate 1-D data that is sorted by indices_of_increasing."""
    data = numpy.asarray(data)
    unsorted = numpy.empty_like(data)
    unsorted[indices_of_increasing] = data
    return unsorted


def plot_transforms(ace_model, fname="ace_transforms.png"):
    """Plot the transforms."""
    if not plt:
        raise ImportError("Cannot plot without the matplotlib package")
    plt.rcParams.update({"font.size": 8})
    plt.figure()
    num_cols = len(ace_model.x) // 2 + 1
    for i in range(len(ace_model.x)):
        plt.subplot(num_cols, 2, i + 1)
        plt.plot(ace_model.x[i], ace_model.x_transforms[i], ".", label=f"Phi {i}")
        plt.xlabel(f"x{i}")
        plt.ylabel(f"phi{i}")
    plt.subplot(num_cols, 2, i + 2)
    plt.plot(ace_model.y, ace_model.y_transform, ".", label="Theta")
    plt.xlabel("y")
    plt.ylabel("theta")
    plt.tight_layout()

    if fname:
        plt.savefig(fname)
        return None
    return plt


def plot_input(ace_model, fname="ace_input.png"):
    """Plot the transforms."""
    if not plt:
        raise ImportError("Cannot plot without the matplotlib package")
    plt.rcParams.update({"font.size": 8})
    plt.figure()
    num_cols = len(ace_model.x) // 2 + 1
    for i in range(len(ace_model.x)):
        plt.subplot(num_cols, 2, i + 1)
        plt.plot(ace_model.x[i], ace_model.y, ".")
        plt.xlabel(f"x{i}")
        plt.ylabel("y")

    plt.tight_layout()

    if fname:
        plt.savefig(fname)
    else:
        plt.show()
