import numpy as np
from sklearn.datasets import load_boston
from db_learn import (compute_weights, mk_Momega, one_sample, SampleX,
                      dim_in_bnd_vec)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

np.random.seed(0)


# Datasets
boston = load_boston()
X = boston.data
y = boston.target

Z = np.hstack((X, y.reshape(-1, 1)))

a_s = np.array([0., 0.])
b_s = np.array([22., 100.])
K = 2

Z_tr, Z_te = train_test_split(Z, test_size=100)

n_s = np.array([200, 100])
lambdas = n_s / n_s.sum()

Z_chp = SampleX(Z_tr, n_s[0], dim=-1, a=a_s[0], b=b_s[0])
Z_exp = SampleX(Z_tr, n_s[1], dim=-1, a=a_s[1], b=b_s[1])
Z_list = [Z_chp, Z_exp]
Z_concat = one_sample(Z_list)


# Compute debiasing weights
def meta_omega(x, k):
    return dim_in_bnd_vec(x, -1, a_s[k], b_s[k])


M_omega = mk_Momega(Z_list, meta_omega)
weights = compute_weights(M_omega, lambdas)


# Train models w/o debiasing
res = np.zeros(6)

reg = LinearRegression()
reg.fit(Z_concat[:, :-1], Z_concat[:, -1])
err = np.mean((reg.predict(Z_te[:, :-1]) - Z_te[:, -1]) ** 2)
res[0] = err

reg_b = LinearRegression()
reg_b.fit(Z_concat[:, :-1], Z_concat[:, -1], sample_weight=weights)
err_b = np.mean((reg_b.predict(Z_te[:, :-1]) - Z_te[:, -1]) ** 2)
res[1] = err_b

reg = SVR(kernel='rbf', gamma='auto')
reg.fit(Z_concat[:, :-1], Z_concat[:, -1])
err = np.mean((reg.predict(Z_te[:, :-1]) - Z_te[:, -1]) ** 2)
res[2] = err

reg_b = SVR(kernel='rbf', gamma='auto')
reg_b.fit(Z_concat[:, :-1], Z_concat[:, -1], sample_weight=weights)
err_b = np.mean((reg_b.predict(Z_te[:, :-1]) - Z_te[:, -1]) ** 2)
res[3] = err_b

reg = RandomForestRegressor(n_estimators=20)
reg.fit(Z_concat[:, :-1], Z_concat[:, -1])
err = np.mean((reg.predict(Z_te[:, :-1]) - Z_te[:, -1]) ** 2)
res[4] = err

reg_b = RandomForestRegressor(n_estimators=20)
reg_b.fit(Z_concat[:, :-1], Z_concat[:, -1], sample_weight=weights)
err_b = np.mean((reg_b.predict(Z_te[:, :-1]) - Z_te[:, -1]) ** 2)
res[5] = err_b

print(res)
