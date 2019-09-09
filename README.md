# Debiased Statistical Learning

## Summary

This Python package contains code to perform debiased Empirical Risk Minimization (ERM). If given a collection of `K` biased training samples `X_k` - as well as the biasing functions omega_k - it is possible to compute debiasing weights such that the minimizer of the weighted criterion behaves similarly to the minimizer a hypothetically unbiased criterion. The main function is `db_weights.compute_weights()`, which takes as input the `M_omega` array of shape `(K, n_max, K)` such that `M_omega[i, j, k] = omega_k(X_{ij})`. It outputs the debiasing weights to be passed to the option `sample_weight` of `scikit-learn`'s predictors.


## Installation
To install the package, simply clone it, and then do:

  `$ pip install -e .`

To check that everything worked, the command

  `$ python -c 'import db_learn'`

should not return any error.

## Use
See the toy example available at `toy_example/toy_example.py`.
