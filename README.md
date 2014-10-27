ace
===

ace is a Python package for performing Alternating Conditional Expectation (ACE) non-parametric 
regressions. It can be used to:

 - build easy-to-evaluate surrogate models of data. For example, if you are studying input 
   parameters to a complex and long-running simulation, you can feed the results of a parameter 
   sweep into ACE to get a model that will instantly give you predictions of results of any 
   combination of input within the parameter range.
    
 - expose interesting and meaningful relations between predictors and responses from complicated 
   data sets. For instance, if you have survey results from 1000 people and you and you want to 
   see how one answer is related to a bunch of others, ACE will help you. 
 
Before this package, the ACE algorithm has only been available in Python by using the rpy2 module 
to load in the acepack package of the R statistical language. This package is a pure-Python 
re-write of the ACE algorithm based on the original publication, using modern software practices. 
This package is slower than the original FORTRAN code, but it is easier to understand. This package 
should be suitable for medium-weight data and as a learning tool. 

It can crunch through a problem with 5 independent variables having 1000 observations each
in on the order of 15 seconds. 

ace also contains a pure-Python implementation of Friedman's SuperSmoother [2], a variable span 
smoother which can be used to smooth scatterplot data. 

Usage
=====
To use ace, you can follow this example problem (from [3]):

```python
from ace import ace

def build_sample_problem(N = 100):
    x = [numpy.array([random.random() * 2.0 - 1.0 for i in range(N)])
         for _i in range(0, 5)]
    noise = numpy.random.standard_normal(N)
    y = numpy.log(4.0 + numpy.sin(4 * x[0]) + numpy.abs(x[1]) + x[2] ** 2 +
                 x[3] ** 3 + x[4] + 0.1 * noise)
    ace_solver = ace.ACESolver()
    ace_solver.specify_data_set(x, y)

    return ace_solver

ace_solver = build_sample_problem()
ace_solver.solve()
```

The results will be stored on the `ace_solver` object. 

History
=======
The ACE algorithm was published in 1985 by Breiman and Friedman [1], and the original FORTRAN 
source code is available from [Friedman's webpage](http://statweb.stanford.edu/~jhf/). 

References
==========
1. L. BREIMAN and J. H. FRIEDMAN, "Estimating optimal transformations for multiple regression and 
   correlation," Journal of the American Statistical Association, 80, 580 (1985). 
   [link](http://www.jstor.org/discover/10.2307/2288477?uid=2&uid=4&sid=21104902100507)

2. J. H. FRIEDMAN and W. STUETZLE, "Smoothing of scatterplots," ORION-003, Stanford
   University, (1982). [link](http://www.slac.stanford.edu/cgi-wrap/getdoc/slac-pub-3013.pdf)

3. D. WANG and M. MURPHY, "Estimating optimal transformations for multiple regression using the 
   ACE algorithm," Journal of Data Science, 2, 329 (2004). 
   [link](http://www.jds-online.com/files/JDS-156.pdf)


