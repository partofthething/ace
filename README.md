ace is an implementation of the Alternating Conditional Expectation (ACE) algorithm [Breiman85]_.

The full documentation, including examples, is hosted at http://partofthething.com/ace.

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

.. [Breiman85] L. BREIMAN and J. H. FRIEDMAN, "Estimating optimal transformations for multiple regression and
   correlation," Journal of the American Statistical Association, 80, 580 (1985).
   `[Link] <http://www.jstor.org/discover/10.2307/2288477?uid=2&uid=4&sid=21104902100507>`_