# Authors: Pierre Laforgue <pierre.laforgue@telecom-paristech.fr>
#
# License: MIT

import numpy as np
from numba import njit


# In this file we detail several useful biasing functions, and ways to sample
# biased training samples.


#########################
#                       #
#   BIASING FUNCTIONS   #
#                       #
#########################


# We now detail several useful biasing functions.


@njit
def norm_in_bnd(x, a, b):
    """Norm between bounds indicator function"""

    z = np.linalg.norm(x)
    sup_a = a < z
    inf_b = z < b
    res = sup_a * inf_b
    return res


@njit
def norm_in_bnd_vec(X, a, b):
    """Norm between bounds indicator function for 2-dimensional X"""

    n = X.shape[0]
    res = np.zeros(n, dtype=np.int64)
    for i in range(n):
        res[i] = norm_in_bnd(X[i, :], a, b)
    return res


@njit
def norm_out_bnd(x, a, b):
    """Norm out of bounds indicator function"""

    z = np.linalg.norm(x)
    inf_a = z < a
    sup_b = b < z
    res = inf_a + sup_b
    return res


@njit
def norm_out_bnd_vec(X, a, b):
    """Norm out of bounds indicator function for 2-dimensional X"""

    n = X.shape[0]
    res = np.zeros(n, dtype=np.int64)
    for i in range(n):
        res[i] = norm_out_bnd(X[i, :], a, b)
    return res


@njit
def dim_in_bnd(x, dim, a, b):
    """Dimension between bounds indicator function"""

    z = x[dim]
    sup_a = a < z
    inf_b = z < b
    res = sup_a * inf_b
    return res


@njit
def dim_in_bnd_vec(X, dim, a, b):
    """Dimension between bounds indicator function for 2-dimensional X"""

    n = X.shape[0]
    res = np.zeros(n, dtype=np.int64)
    for i in range(n):
        res[i] = dim_in_bnd(X[i, :], dim, a, b)
    return res


@njit
def dim_out_bnd(x, dim, a, b):
    """Dimension out of bounds indicator function"""

    z = x[dim]
    inf_a = z < a
    sup_b = b < z
    res = inf_a + sup_b
    return res


@njit
def dim_out_bnd_vec(X, dim, a, b):
    """Dimension out of bounds indicator function for 2-dimensional X"""

    n = X.shape[0]
    res = np.zeros(n, dtype=np.int64)
    for i in range(n):
        res[i] = dim_out_bnd(X[i, :], dim, a, b)
    return res


@njit
def dim_in_set(x, dim, set_):
    """Dimension in particular set of values indicator function"""

    z = x[dim]
    res = 0
    for item in set_:
        if z == item:
            res = 1
    return res


@njit
def dim_in_set_vec(X, dim, set_):
    """Dimension in particular set of values indicator function for 2d X"""

    n = X.shape[0]
    res = np.zeros(n, dtype=np.int64)
    for i in range(n):
        res[i] = dim_in_set(X[i, :], dim, set_)
    return res


#####################################
#                                   #
#   CONSTRAINED GAUSSIAN DATASETS   #
#                                   #
#####################################


# The following functions combine Gaussian random generation and the previously
# defined biasing functions to generate biased Gaussian samples.


@njit
def Gauss_norm_in_bnd(n, d, a, b):
    """Sample from Gaussian r.v. constrained by norm_in_bnd"""

    X = np.zeros((n, d), dtype=np.float64)
    count = 0
    while count < n:
        x = np.random.randn(d)
        if norm_in_bnd(x, a, b,):
            X[count, :] = x.copy()
            count += 1
    return X


@njit
def Gauss_norm_out_bnd(n, d, a, b):
    """Sample from Gaussian r.v. constrained by norm_out_bnd"""

    X = np.zeros((n, d), dtype=np.float64)
    count = 0
    while count < n:
        x = np.random.randn(d)
        if norm_out_bnd(x, a, b,):
            X[count, :] = x.copy()
            count += 1
    return X


@njit
def Gauss_dim_in_bnd(n, d, dim, a, b):
    """Sample from Gaussian r.v. constrained by dim_in_bnd"""

    X = np.zeros((n, d), dtype=np.float64)
    count = 0
    while count < n:
        x = np.random.randn(d)
        if dim_in_bnd(x, dim, a, b,):
            X[count, :] = x.copy()
            count += 1
    return X


@njit
def Gauss_dim_out_bnd(n, d, dim, a, b):
    """Sample from Gaussian r.v. constrained by dim_out_bnd"""

    X = np.zeros((n, d), dtype=np.float64)
    count = 0
    while count < n:
        x = np.random.randn(d)
        if dim_out_bnd(x, dim, a, b,):
            X[count, :] = x.copy()
            count += 1
    return X


def Gauss(n, d, dim='norm', a=0., b=1., in_=True):
    """Sample from Gaussian r.v. constrained by any function

    Parameters
    ----------
    n: int
       Number of observations to sample

    d: int
       Dimension of the observations to sample

    dim: str, default='norm'
         Perform bias on norm or on dimension (to be specified by an int)

    a: float, default=0.
       Lower bound of the bias

    b: float, default=1.
       Upper bound of the bias

    in_: bool, default=True
         If True, sample Gaussian observations that satisfy norm (or specified)
         dimension between a and b. Outside a and b otherwise.

    Returns
    -------
    X: array of shape (n, d)
       Biased Gaussian sample
    """

    if dim == 'norm':
        if in_:
            X = Gauss_norm_in_bnd(n, d, a, b)
        else:
            X = Gauss_norm_out_bnd(n, d, a, b)

    else:
        if in_:
            X = Gauss_dim_in_bnd(n, d, dim, a, b)
        else:
            X = Gauss_dim_out_bnd(n, d, dim, a, b)

    return X


#############################################
#                                           #
#   SAMPLE WITH CONSTRAINTS FROM DATASETS   #
#                                           #
#############################################


# Another interesting feature is to sample from an existing dataset, with
# respect to a biasing function


@njit
def SampleX_norm_in_bnd(X, n, a, b):
    """Sample from X constrained by norm_in_bnd"""

    n_max = norm_in_bnd_vec(X, a, b).sum()
    if n > n_max:
        print('Not enough data verifying condition, chosen instead:')
        print(n_max)
        return SampleX_norm_in_bnd(X, n_max, a, b)

    else:
        d = X.shape[1]
        X_bias = np.zeros((n, d), dtype=np.float64)
        count = 0
        i = 0

        while count < n:
            x = X[i, :]
            if norm_in_bnd(x, a, b,):
                X_bias[count, :] = x.copy()
                count += 1
            i += 1

        return X_bias


@njit
def SampleX_norm_out_bnd(X, n, a, b):
    """Sample from X constrained by norm_out_bnd"""

    n_max = norm_out_bnd_vec(X, a, b).sum()
    if n > n_max:
        print('Not enough data verifying condition, chosen instead:')
        print(n_max)
        return SampleX_norm_out_bnd(X, n_max, a, b)

    else:
        d = X.shape[1]
        X_bias = np.zeros((n, d), dtype=np.float64)
        count = 0
        i = 0

        while count < n:
            x = X[i, :]
            if norm_out_bnd(x, a, b,):
                X_bias[count, :] = x.copy()
                count += 1
            i += 1

        return X_bias


@njit
def SampleX_dim_in_bnd(X, n, dim, a, b):
    """Sample from X constrained by dim_in_bnd"""

    n_max = dim_in_bnd_vec(X, dim, a, b).sum()
    if n > n_max:
        print('Not enough data verifying condition, chosen instead:')
        print(n_max)
        return SampleX_dim_in_bnd(X, n_max, dim, a, b)

    else:
        d = X.shape[1]
        X_bias = np.zeros((n, d), dtype=np.float64)
        count = 0
        i = 0

        while count < n:
            x = X[i, :]
            if dim_in_bnd(x, dim, a, b,):
                X_bias[count, :] = x.copy()
                count += 1
            i += 1

        return X_bias


@njit
def SampleX_dim_out_bnd(X, n, dim, a, b):
    """Sample from X constrained by dim_out_bnd"""

    n_max = dim_out_bnd_vec(X, dim, a, b).sum()
    if n > n_max:
        print('Not enough data verifying condition, chosen instead:')
        print(n_max)
        return SampleX_dim_out_bnd(X, n_max, dim, a, b)

    else:
        d = X.shape[1]
        X_bias = np.zeros((n, d), dtype=np.float64)
        count = 0
        i = 0

        while count < n:
            x = X[i, :]
            if dim_out_bnd(x, dim, a, b,):
                X_bias[count, :] = x.copy()
                count += 1
            i += 1

        return X_bias


@njit
def SampleX_dim_in_set(X, n, dim, set_):
    """Sample from X constrained by dim_in_bnd"""

    n_max = dim_in_set_vec(X, dim, set_).sum()
    if n > n_max:
        print('Not enough data verifying condition, chosen instead:')
        print(n_max)
        return SampleX_dim_in_set(X, n_max, dim, set_)

    else:
        d = X.shape[1]
        X_bias = np.zeros((n, d), dtype=np.float64)
        count = 0
        i = 0

        while count < n:
            x = X[i, :]
            if dim_in_set(x, dim, set_):
                X_bias[count, :] = x.copy()
                count += 1
            i += 1

        return X_bias


def SampleX(X, n, dim='norm', a=0., b=1., in_=True):
    """Sample from X constrained by any function

    Parameters
    ----------
    X: array of shape (n_obs, d)
       Original dataset from which observations are sampled

    n: int
       Number of observations to sample

    dim: str, default='norm'
         Perform bias on norm or on dimension (to be specified by an int)

    a: float, default=0.
       Lower bound of the bias

    b: float, default=1.
       Upper bound of the bias

    in_: bool, default=True
         If True, sample observations that satisfy norm (or specified)
         dimension between a and b. Outside a and b otherwise

    Returns
    -------
    X_bias: array of shape (n, d)
            Set of observations sampled from X with specified bias
    """

    X2 = X.copy()
    np.random.shuffle(X2)

    if dim == 'norm':
        if in_:
            X_bias = SampleX_norm_in_bnd(X2, n, a, b)
        else:
            X_bias = SampleX_norm_out_bnd(X2, n, a, b)

    else:
        if isinstance(in_, np.bool):
            if in_:
                X_bias = SampleX_dim_in_bnd(X2, n, dim, a, b)
            else:
                X_bias = SampleX_dim_out_bnd(X2, n, dim, a, b)

        else:
            X_bias = SampleX_dim_in_set(X2, n, dim, in_)

    return X_bias
