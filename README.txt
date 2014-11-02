ace is an implementation of the Alternating Conditional Expectation (ACE) algorithm [Breiman85]_,
which can be used to find otherwise difficult-to-find relationships between predictors
and responses and as a multivariate regression tool.

The full documentation is hosted at http://partofthething.com/ace.
The source code, bug tracker, etc., can be found at: https://github.com/partofthething/ace

What is it?
===========
ACE can be used for a variety of purposes. With it, you can:

 - build easy-to-evaluate surrogate models of data. For example, if you are optimizing input
   parameters to a complex and long-running simulation, you can feed the results of a parameter
   sweep into ACE to get a model that will instantly give you predictions of results of any
   combination of input within the parameter range.

 - expose interesting and meaningful relations between predictors and responses from complicated
   data sets. For instance, if you have survey results from 1000 people and you and you want to
   see how one answer is related to a bunch of others, ACE will help you.

The fascinating thing about ACE is that it is a *non-parametric* multivariate regression
tool. This means that it doesn't make any assumptions about the functional form of the data.
You may be used to fitting polynomials or lines to data. Well, ACE doesn't do that. It
uses an iteration with a variable-span scatterplot smoother (implementing local least
squares estimates) to figure out the structure of your data. As you'll see, that
turns out to be a powerful difference.

Installing it
=============
On Linux::

	sudo pip install ace

On Windows::

	pip install ace


Using it
========
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

More details
============
This implementation of ACE isn't as fast as the original FORTRAN version, but it can
still crunch through a problem with 5 independent variables having 1000 observations each
in on the order of 15 seconds. Not bad.

ace also contains a pure-Python implementation of Friedman's SuperSmoother [Friedman82]_,
the variable-span smoother mentioned above. This can be useful on its own
for smoothing scatterplot data.

References
==========
.. [Breiman85] L. BREIMAN and J. H. FRIEDMAN, "Estimating optimal transformations for multiple regression and
   correlation," Journal of the American Statistical Association, 80, 580 (1985).
   `[PDF at JSTOR] <http://www.jstor.org/discover/10.2307/2288477?uid=2&uid=4&sid=21104902100507>`_

.. [Friedman82] J. H. FRIEDMAN and W. STUETZLE, "Smoothing of scatterplots," ORION-003, Stanford
   University, (1982). `[PDF at Stanford] <http://www.slac.stanford.edu/cgi-wrap/getdoc/slac-pub-3013.pdf>`_