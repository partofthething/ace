
ace documentation
=================

ace is an implementation of the Alternating Conditional Expectation (ACE) algorithm [Breiman85]_.
The code for this project, as well as the issue tracker, etc. is
`hosted on GitHub <https://github.com/partofthething/ace>`_.

.. toctree::
   :maxdepth: 2

What does the ACE algorithm do?
===============================
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

This implementation of ACE isn't as fast as the original FORTRAN version, but it can
still crunch through a problem with 5 independent variables having 1000 observations each
in on the order of 15 seconds. Not bad.

ace also contains a pure-Python implementation of Friedman's SuperSmoother [Friedman82]_,
the variable-span smoother mentioned above. This can be useful on its own
for smoothing scatterplot data.

Installation
============
To install ace from the Python Package Index, simply run::

	pip install ace

To install ace directly from source, run::

	git clone git@github.com:partofthething/ace.git
	cd ace
	python setup.py install

You can verify that the installation completed successfully by running the automated test
suite::

	python -m unittest discover -bv

Using ACE
=========
The simplest way to run ACE and get results is to use the
:py:class:`Model <ace.model.Model>` object. But first you need some data. Let's
assume you have two independent variables and one dependent variable and that you
are storing data in the ``x`` and ``y`` variables, like this::

	>>> x
	[(0.1,0.2,0.0.05,...),
	 (1,5,3,6,...)]
	>>> y
	[10.0,20.0,15.0,...]

Then you an run ACE on it like this::

	from ace import model

	myace = model.Model()
	myace.build_model_from_xy(x, y) # runs ACE

Now you have a full ACE regression built up. You can now
evaluate ``y`` as a function of any values of ``x`` using
:py:meth:`eval <ace.model.Model.eval>`, like this::

	myace.eval[(0.11, 5.4)]

You can also plot your transforms or export the results::

	from ace import ace
	ace.plot_transforms(myace, fname = 'mytransforms.pdf')
	myace.ace.write_transforms_to_file(fname = 'mytransforms.txt')

Note that you could alternatively have loaded your data from a whitespace delimited
text file::

	myace.build_model_from_txt(fname = 'myinput.txt')

Sample ACE Problems
===================
Several sample problems from public literature are provided in the :py:mod:`ace.samples`
subpackage. The one from [Wang04]_ is particularly good at demonstrating the power
of the ACE algorithm. It starts by building test data using the following formula:

.. math::

	y(X) = \text{log}\left(4 + \text{sin}(4 X_0) + |X_1| + X_2^2 + X_3^3 + X_4 + 0.1\epsilon\right)

where :math:`\epsilon` is sampled from standard normal distribution. The input data (N=200) is
shown below. As you can see, it's pretty ugly, and it'd be difficult to understand the
underlying structure by doing normal regressions.

.. image:: _static/ace_input_wang04.png
	   :alt: Plot of the input data, which is all over the place

But go ahead and try running ACE on it, like this::

	from ace.samples import wang04
	wang04.run_wang04()

This will produce resulting transforms that look like the ones in the figure below. As you can
see, ACE performed surprisingly well at extracting the underlying structure. You can clearly
see the sine wave, the absolute value, the cubic, etc. It's fabulous!

.. image:: _static/ace_transforms_wang04.png
	:alt: Plot of the output transforms, which clearly show the underlying structure

History
=======
The ACE algorithm was published in 1985 by Breiman and Friedman [Breiman85]_, and the original
FORTRAN source code is available from `Friedman's webpage <http://statweb.stanford.edu/~jhf/>`_.

Motivation
==========
Before this package, the ACE algorithm has only been available in Python by using the rpy2 module
to load in the acepack package of the R statistical language. This package is a pure-Python
re-write of the ACE algorithm based on the original publication, using modern software practices.
This package is slower than the original FORTRAN code, but it is easier to understand. This package
should be suitable for medium-weight data and as a learning tool.

For the record, it is also quite easy to run the original FORTRAN code in Python using f2py.

About the Author
================
This package was originated by Nick Touran, a nuclear engineer specializing in reactor physics.
He was exposed to ACE by his thesis advisor, Professor John Lee, and used it in his
Ph.D. dissertation to evaluate objective functions in a multidisciplinary
design optimization study of nuclear reactor cores [Touran12]_.

License
=======
This package is released under the MIT License, reproduced
`here <https://github.com/partofthething/ace/blob/master/LICENSE>`_.

References
==========
.. [Breiman85] L. BREIMAN and J. H. FRIEDMAN, "Estimating optimal transformations for multiple regression and
   correlation," Journal of the American Statistical Association, 80, 580 (1985).
   `[Link] <http://www.jstor.org/discover/10.2307/2288477?uid=2&uid=4&sid=21104902100507>`_

.. [Friedman82] J. H. FRIEDMAN and W. STUETZLE, "Smoothing of scatterplots," ORION-003, Stanford
   University, (1982). `[Link] <http://www.slac.stanford.edu/cgi-wrap/getdoc/slac-pub-3013.pdf>`_

.. [Wang04] D. WANG and M. MURPHY, "Estimating optimal transformations for multiple regression using the
   ACE algorithm," Journal of Data Science, 2, 329 (2004).
   `[Link] <http://www.jds-online.com/files/JDS-156.pdf>`_

..  [Touran12] N. TOURAN, "A Modal Expansion Equilibrium Cycle Perturbation Method for
    Optimizing High Burnup Fast Reactors," Ph.D. dissertation, Univ. of Michigan, (2012).
	`[The Thesis] <http://deepblue.lib.umich.edu/bitstream/handle/2027.42/95981/ntouran_1.pdf?sequence=1>`_


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

