import numpy as np
from scipy.optimize import root, fmin_l_bfgs_b


# The main function of this file is "compute_weights", that takes as inputs the
# M_omega array of shape (K, n_max, K) such that M_omega[i, j, k] =
# omega_k(X_ij), with nan values wherever X_ij is not defined, and the lambdas.
# It returns the debiasing weights to be used in every learning task.


##################
#                #
#   COMPUTE WS   #
#                #
##################


# The Ws are solutions to system (8) in Laforgue & Clemencon 2019. Solving the
# system is equivalent to minimizing the convex function D in u, up to the
# change of variable u = log(lambda / W). The following functions give several
# ways to solve the system, and find the Ws, provided the M_omega array.


def Ws_from_u(u, lambdas):
    """Recover Ws from Us"""

    Ws = lambdas * np.exp(-u)
    Ws /= Ws[-1]
    return Ws


def D(u, M_omega, lambdas):
    """Objective function"""

    eu = np.exp(u)
    sum_ = np.einsum('k,ijk->ij', eu, M_omega)
    log_sum_ = np.log(sum_)

    res = np.nanmean(log_sum_)
    res -= np.dot(lambdas, u)

    return res


def D2(u_minus, M_omega, lambdas):
    """Objective function when last component is fixed"""

    u = np.hstack((u_minus, np.log(lambdas[-1])))
    res = D(u, M_omega, lambdas)
    return res


def K_(u, M_omega, lambdas):
    """Gradient function"""

    K, n_max = M_omega.shape[0], M_omega.shape[1]
    n = K * n_max - np.isnan(M_omega[:, :, 0]).sum()
    eu = np.exp(u)

    eu_Omega = eu[None, None, :] * M_omega
    eu_Omega_s = np.sum(eu_Omega, axis=2)

    Y = eu_Omega / eu_Omega_s[:, :, None]
    res = 1. / n * np.nansum(Y, axis=(0, 1)) - lambdas

    return res


def K2(u_minus, M_omega, lambdas):
    """Gradient function when last component is fixed"""

    u = np.hstack((u_minus, np.log(lambdas[-1])))
    res = K_(u, M_omega, lambdas)
    return res[:-1]


def compute_Ws_RM(M_omega, lambdas, n_epoch=1000, lr=1.):
    """Compute Ws by Robbins-Monro algorithm"""

    # init
    K = M_omega.shape[0]
    u_minus = np.random.randn(K - 1)

    for epoch in range(n_epoch):
        K_value = K2(u_minus, M_omega, lambdas)
        u_minus -= lr * K_value

    u = np.hstack((u_minus, np.log(lambdas[-1])))
    Ws = Ws_from_u(u, lambdas)
    return Ws


def compute_Ws_root(M_omega, lambdas):
    """Compute Ws by rooting K2"""

    K = M_omega.shape[0]
    u0_minus = np.random.randn(K - 1)

    sol = root(K2, u0_minus, args=(M_omega, lambdas))

    u = np.hstack((sol.x, np.log(lambdas[-1])))
    Ws = Ws_from_u(u, lambdas)

    return Ws


def compute_Ws_lbfgs(M_omega, lambdas):
    """Compute Ws by minimizing D2 via lbfgs"""

    K = M_omega.shape[0]
    u0_minus = np.random.randn(K - 1)

    sol_bfgs, _, _ = fmin_l_bfgs_b(D2, u0_minus, args=(M_omega, lambdas),
                                   fprime=K2, approx_grad=0, pgtol=1e-10)

    u = np.hstack((sol_bfgs, np.log(lambdas[-1])))
    Ws = Ws_from_u(u, lambdas)

    return Ws


def compute_Ws(M_omega, lambdas):
    """Compute Ws (use successive techniques to be sure to converge)"""

    Ws = compute_Ws_RM(M_omega, lambdas)
    u = np.log(lambdas / Ws)
    n = np.linalg.norm(K_(u, M_omega, lambdas))

    if n > 1e-6:
        Ws = compute_Ws_root(M_omega, lambdas)
        u = np.log(lambdas / Ws)
        n = np.linalg.norm(K_(u, M_omega, lambdas))

        if n > 1e-6:
            Ws = compute_Ws_lbfgs(M_omega, lambdas)
            u = np.log(lambdas / Ws)
            n = np.linalg.norm(K_(u, M_omega, lambdas))

            if n > 1e-6:
                print('Convergence Warning, K_norm: %2.e' % n)

    return Ws


############################
#                          #
#   RECOVER OMEGA FROM W   #
#                          #
############################


# Recovering the Omegas necessitates calculating the normalizing constant of
# equation (1. 19) in Gill et al. 1988.


def normalzing_constant(Ws, M_omega, lambdas):
    """Compute normalizing constant"""

    b = lambdas / Ws
    C = np.einsum('k,ijk->ij', b, M_omega)
    D = 1. / C
    res = np.nanmean(D)
    return res


def Omegas_from_Ws(Ws, M_omega, lambdas):
    """Compute Omegas from Ws"""

    c = normalzing_constant(Ws, M_omega, lambdas)
    Omegas = Ws / c
    return Omegas


def compute_Omegas(M_omega, lambdas):
    """Compute Omegas directly from M_omega"""

    Ws = compute_Ws(M_omega, lambdas)
    Omegas = Omegas_from_Ws(Ws, M_omega, lambdas)
    return Omegas


#######################
#                     #
#   COMPUTE WEIGHTS   #
#                     #
#######################


# Using the debiased empirical law boils down to compute the individual weights
# of equation (5) in Laforgue & Clemencon 2019.


def weights_from_Omegas(Omegas, M_omega, lambdas):
    """Compute individual weights"""

    b = lambdas / Omegas
    C = np.einsum('k,ijk->ij', b, M_omega)
    D = 1. / C
    res = D.ravel()
    res = res[~np.isnan(res)]
    return res


def compute_weights(M_omega, lambdas):
    """Compute individual weights directly from M_omega"""

    Omegas = compute_Omegas(M_omega, lambdas)
    weights = weights_from_Omegas(Omegas, M_omega, lambdas)
    return weights


#####################
#                   #
#  COMPUTE M_OMEGA  #
#                   #
#####################


# Finally, we detail how to compute the M_omega array from a list of (biased)
# samples and a meta biasing function such that meta_omega(x, k) = omega_k(x).


def mk_Momega(X_list, meta_omega):
    """Create M_omega array such that M_omega[i, j, k] = omega_k(X_ij)"""

    K = len(X_list)
    n_s = [X.shape[0] for X in X_list]
    n_max = np.max(n_s)
    M_omega = np.full((K, n_max, K), np.nan)
    for i in range(K):
        for k in range(K):
            M_omega[i, :n_s[i], k] = meta_omega(X_list[i], k)
    return M_omega


def one_sample(X_list):
    """Concatenate all samples"""

    K = len(X_list)
    X = X_list[0].copy()
    for k in range(K - 1):
        X = np.vstack((X, X_list[k + 1]))
    return X
