# fatwalrus
Code for laying out on the rocks all day with the homies and doing Bayesian statistics.

![alt tag](https://github.com/aschein/fatwalrus/blob/master/IMG_7763.jpg)

## What's included:

* [bessel.pyx](https://github.com/aschein/fatwalrus/blob/master/src/bessel.pyx): Implements rejection sampling for the Bessel distribution.
* [mcmc_model.pyx](https://github.com/aschein/fatwalrus/blob/master/src/mcmc_model.pyx): Implements Cython interface for MCMC models.  Inherited by pgds.pyx.
* [sample.pyx](https://github.com/aschein/fatwalrus/blob/master/src/sample.pyx): Implements fast Cython method for sampling various distributions.
* [slice_sample.pyx](https://github.com/aschein/fatwalrus/blob/master/src/sample.pyx): Implements general slice-sampling interface.

## Dependencies:

* numpy
* scipy
* argparse
* path
* scikit-learn
* cython
* GSL





