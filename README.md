# fatwalrus
Code for laying out on the rocks all day with the homies.


![alt tag](https://github.com/aschein/fatwalrus/blob/master/IMG_7763.jpg)


## What's included:

* [pgds.pyx](https://github.com/aschein/pgds/blob/master/src/pgds.pyx): The main code file.  Implements Gibbs sampling inference for PGDS.
* [mcmc_model.pyx](https://github.com/aschein/pgds/blob/master/src/mcmc_model.pyx): Implements Cython interface for MCMC models.  Inherited by pgds.pyx.
* [sample.pyx](https://github.com/aschein/pgds/blob/master/src/sample.pyx): Implements fast Cython method for sampling various distributions.
* [lambertw.pyx](https://github.com/aschein/pgds/blob/master/src/lambertw.pyx): Code for computing the Lambert-W function.
* [Makefile](https://github.com/aschein/pgds/blob/master/src/Makefile): Makefile (cd into this directoy and type 'make' to compile).

## Dependencies:

* numpy
* scipy
* argparse
* path
* scikit-learn
* cython
* GSL
