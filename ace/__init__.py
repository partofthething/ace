"""
ace is a multivariate regression tool that solves alternating conditional expectations.

See full documentation at http://partofthething.com/ace

To use, get some sample data::

    from ace.samples import wang04
    x, y = wang04.build_sample_ace_problem_wang04(N=200)

and run::

    from ace import model
    myace = model.Model()
    myace.build_model_from_xy(x, y)
    myace.eval([0.1, 0.2, 0.5, 0.3, 0.5])

For some plotting (matplotlib required), try::

    from ace import ace
    ace.plot_transforms(myace, fname = 'mytransforms.pdf')
    myace.ace.write_transforms_to_file(fname = 'mytransforms.txt')

"""

import os.path
from pkg_resources import get_distribution, DistributionNotFound

def _get_version():
    """
    Groan single sourcing versions is a huge pain.

    For now we have to manually sync it between here and setup.py (for doc build)

    https://packaging.python.org/guides/single-sourcing-package-version/
    """
    try:
        _dist = get_distribution('ace')
        # Normalize case for Windows systems
        _dist_loc = os.path.normcase(_dist.location)  # pylint: disable=no-member
        _here = os.path.normcase(__file__)
        if not _here.startswith(os.path.join(_dist_loc, 'ace')):
            # not installed, but there is another version that *is*
            raise DistributionNotFound
    except DistributionNotFound:
        version = '0.3.2'
    else:
        version = _dist.version  # pylint: disable=no-member

    return version

__version__ = _get_version()
