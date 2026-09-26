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

__version__ = "0.4.0"
