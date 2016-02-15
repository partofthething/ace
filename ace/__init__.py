"""
ace is a multivariate regression tool that solves alternating conditional expectations

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
def _get_version():
    from pkg_resources import get_distribution, DistributionNotFound
    import os.path
    try:
        _dist = get_distribution('ace')
        # Normalize case for Windows systems
        _dist_loc = os.path.normcase(_dist.location)
        _here = os.path.normcase(__file__)
        if not _here.startswith(os.path.join(_dist_loc, 'ace')):
            # not installed, but there is another version that *is*
            raise DistributionNotFound
    except DistributionNotFound:
        version = 'Please install ace with setup.py'
    else:
        version = _dist.version

    return version

__version__ = _get_version()

from . import ace
from . import model
from . import smoother
from . import supersmoother
from . import samples
from . import tests


