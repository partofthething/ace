r"""
The Alternating Condtional Expectation (ACE) algorithm.

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

import numpy
try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None

from .supersmoother import SuperSmoother
from .smoother import perform_smooth


MAX_OUTERS = 200


class ACESolver(object):  # pylint: disable=too-many-instance-attributes
    """The Alternating Conditional Expectation algorithm to perform regressions."""

    def __init__(self):
        """Solver constructor."""
        self._last_inner_error = float('inf')
        self._last_outer_error = float('inf')
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
        while self._outer_error_is_decreasing() and self._outer_iters < MAX_OUTERS:
            print('* Starting outer iteration {0:03d}. Current err = {1:12.5E}'
                  ''.format(self._outer_iters, self._last_outer_error))
            self._iterate_to_update_x_transforms()
            self._update_y_transform()
            self._outer_iters += 1

    def _initialize(self):
        """Set up and normalize initial data once input data is specified."""
        self.y_transform = self.y - numpy.mean(self.y)
        self.y_transform /= numpy.std(self.y_transform)
        self.x_transforms = [numpy.zeros(len(self.y)) for _xi in self.x]
        self._compute_sorted_indices()

    def _compute_sorted_indices(self):
        """
        Sort data from the perspective of each column.

        if self._x[0][3] is the 9th-smallest value in self._x[0], then  _xi_sorted[3] = 8

        We only have to sort the data once.
        """
        sorted_indices = []
        for to_sort in [self.y] + self.x:
            data_w_indices = [(val, i) for (i, val) in enumerate(to_sort)]
            data_w_indices.sort()
            sorted_indices.append([i for val, i in data_w_indices])
        # save in meaningful variable names
        self._yi_sorted = sorted_indices[0]  # list (like self.y)
        self._xi_sorted = sorted_indices[1:]  # list of lists (like self.x)

    def _outer_error_is_decreasing(self):
        """Return True if outer iteration error is decreasing."""
        is_decreasing, self._last_outer_error = self._error_is_decreasing(self._last_outer_error)
        return is_decreasing

    def _error_is_decreasing(self, last_error):
        """Return True if current error is less than last_error."""
        current_error = self._compute_error()
        is_decreasing = current_error < last_error
        return is_decreasing, current_error

    def _compute_error(self):
        """Compute unexplained error."""
        sum_x = sum(self.x_transforms)
        err = sum((self.y_transform - sum_x) ** 2) / len(sum_x)
        return err

    def _iterate_to_update_x_transforms(self):
        """Perform the inner iteration."""
        self._inner_iters = 0
        self._last_inner_error = float('inf')
        while self._inner_error_is_decreasing():
            print('  Starting inner iteration {0:03d}. Current err = {1:12.5E}'
                  ''.format(self._inner_iters, self._last_inner_error))
            self._update_x_transforms()
            self._inner_iters += 1

    def _inner_error_is_decreasing(self):
        is_decreasing, self._last_inner_error = self._error_is_decreasing(self._last_inner_error)
        return is_decreasing

    def _update_x_transforms(self):
        """
        Compute a new set of x-transform functions phik.

        phik(xk) = theta(y) - sum of phii(xi) over i!=k

        This is the first of the eponymous conditional expectations. The conditional
        expectations are computed using the SuperSmoother.
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
            updated_x_transform_smooth = smoother.smooth_result
            updated_x_transform_smooth -= numpy.mean(updated_x_transform_smooth)

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
        smooth = perform_smooth(y_sorted, sum_of_x_transformations_choppy,
                                smoother_cls=self._smoother_cls)
        sum_of_x_transformations_smooth = smooth.smooth_result

        sum_of_x_transformations_smooth -= numpy.mean(sum_of_x_transformations_smooth)
        sum_of_x_transformations_smooth /= numpy.std(sum_of_x_transformations_smooth)

        # unsort to save in the original data
        self.y_transform = unsort_vector(sum_of_x_transformations_smooth, sorted_data_indices)

    def write_input_to_file(self, fname='ace_input.txt'):
        """Write y and x values used in this run to a space-delimited txt file."""
        self._write_columns(fname, self.x, self.y)

    def write_transforms_to_file(self, fname='ace_transforms.txt'):
        """Write y and x transforms used in this run to a space-delimited txt file."""
        self._write_columns(fname, self.x_transforms, self.y_transform)

    def _write_columns(self, fname, xvals, yvals):  # pylint: disable=no-self-use
        with open(fname, 'w') as output_file:
            alldata = [yvals] + xvals
            for datai in zip(*alldata):
                yline = '{0: 15.9E} '.format(datai[0])
                xline = ' '.join(['{0: 15.9E}'.format(xii) for xii in datai[1:]])
                output_file.write(''.join([yline, xline, '\n']))


def sort_vector(data, indices_of_increasing):
    """Permutate 1-d data using given indices."""
    return numpy.array([data[i] for i in indices_of_increasing])


def unsort_vector(data, indices_of_increasing):
    """Upermutate 1-D data that is sorted by indices_of_increasing."""
    return numpy.array([data[indices_of_increasing.index(i)] for i in range(len(data))])


def plot_transforms(ace_model, fname='ace_transforms.png'):
    """Plot the transforms."""
    if not plt:
        raise ImportError('Cannot plot without the matplotlib package')
    plt.rcParams.update({'font.size': 8})
    plt.figure()
    num_cols = len(ace_model.x) // 2 + 1
    for i in range(len(ace_model.x)):
        plt.subplot(num_cols, 2, i + 1)
        plt.plot(ace_model.x[i], ace_model.x_transforms[i], '.', label='Phi {0}'.format(i))
        plt.xlabel('x{0}'.format(i))
        plt.ylabel('phi{0}'.format(i))
    plt.subplot(num_cols, 2, i + 2)  # pylint: disable=undefined-loop-variable
    plt.plot(ace_model.y, ace_model.y_transform, '.', label='Theta')
    plt.xlabel('y')
    plt.ylabel('theta')
    plt.tight_layout()

    if fname:
        plt.savefig(fname)
        return None
    return plt

def plot_input(ace_model, fname='ace_input.png'):
    """Plot the transforms."""
    if not plt:
        raise ImportError('Cannot plot without the matplotlib package')
    plt.rcParams.update({'font.size': 8})
    plt.figure()
    num_cols = len(ace_model.x) // 2 + 1
    for i in range(len(ace_model.x)):
        plt.subplot(num_cols, 2, i + 1)
        plt.plot(ace_model.x[i], ace_model.y, '.')
        plt.xlabel('x{0}'.format(i))
        plt.ylabel('y')

    plt.tight_layout()

    if fname:
        plt.savefig(fname)
    else:
        plt.show()
