import numpy as np
import tepimport
import matplotlib.pyplot as plt
import time
from math import floor

from src.sfamanopt import incmssfa, ssfa, mssfa, sfa
from SlowFeatureAnalysis.src.sfafd import incsfa, rsfa
from sklearn.decomposition import SparsePCA
import warnings
warnings.filterwarnings("ignore")

n_lags = 10
n_J = 11
n_iters_lag = 5
n_iters_J = 5

const_J = 10  # For runtimes across m vals
const_lag = 2  # For runtimes across J vals

lag_times = np.zeros((8, n_lags, n_iters_lag))
J_times = np.zeros((8, n_J, n_iters_J))

for lag in range(n_lags):
    print(f'Lag={lag}')
    X, T0, T4, T5, T10 = tepimport.import_tep_sets(lagged_samples=lag)
    m, n = X.shape
    X_mean = np.mean(X, axis=1).reshape((-1, 1))
    X = X - X_mean
    X_std = np.std(X, axis=1).reshape((-1, 1))
    X = X / X_std

    for iteration in range(n_iters_lag):
        """Import Data"""
        sfa_object = sfa.SFA()
        ssfa_object = ssfa.SSFA()
        spca_object = SparsePCA(n_components=const_J, max_iter=500, tol=1e-6)
        mssfa_object = mssfa.MSSFA("chol", "l1")
        rsfa_object = rsfa.RSFA(m, const_J, const_J, L=2, conv_tol=0)
        incsfa_object = incsfa.IncSFA(m, const_J, const_J, L=2, conv_tol=0)
        incsfa_svd_object = incsfa.IncSFA(m, const_J, const_J, L=2, conv_tol=0)
        incmssfa_object = incmssfa.IncMSSFA()

        print("    SFA")
        sfa_time = time.time()
        sfa_object.run(X, const_J)
        sfa_time = time.time() - sfa_time

        print("    SSFA")
        ssfa_time = time.time()
        ssfa_object.run(X, const_J)
        ssfa_time = time.time() - ssfa_time

        print("    SPCA")
        spca_time = time.time()
        spca_object.fit(X.T)
        spca_time = time.time() - spca_time

        print("    MSSFA")
        mssfa_time = time.time()
        mssfa_object.run(X, const_J)
        mssfa_time = time.time() - mssfa_time

        print("    RSFA")
        rsfa_time = time.time()
        for i in range(n):
            rsfa_object.add_data(X[:, i])
        rsfa_time = time.time() - rsfa_time

        print("    IncSFA")
        incsfa_time = time.time()
        for i in range(n):
            incsfa_object.add_data(X[:, i], update_monitors=False,
                                   use_svd_whitening=False)
        incsfa_time = time.time() - incsfa_time

        print("    IncSFA SVD")
        incsfa_svd_time = time.time()
        for i in range(n):
            incsfa_svd_object.add_data(X[:, i], update_monitors=False,
                                       use_svd_whitening=True)
        incsfa_svd_time = time.time() - incsfa_svd_time

        print("    IncMSSFA")
        incmssfa_time = time.time()
        incmssfa_object.run(X, const_J, L=2)
        incmssfa_time = time.time() - incmssfa_time

        lag_times[0, lag, iteration] = sfa_time
        lag_times[1, lag, iteration] = ssfa_time
        lag_times[2, lag, iteration] = spca_time
        lag_times[3, lag, iteration] = mssfa_time
        lag_times[4, lag, iteration] = rsfa_time
        lag_times[5, lag, iteration] = incsfa_time
        lag_times[6, lag, iteration] = incsfa_svd_time
        lag_times[7, lag, iteration] = incmssfa_time

lag_times = lag_times.mean(axis=2)
x_vals = [33 * (i + 1) for i in range(n_lags)]

fig_batch, ax_batch = plt.subplots()
ax_batch.plot(x_vals, lag_times[0, :], '--o', label='SFA')
ax_batch.plot(x_vals, lag_times[1, :], '--s', label='SSFA')
ax_batch.plot(x_vals, lag_times[2, :], '--^', label='SPCA')
ax_batch.plot(x_vals, lag_times[3, :], '--P', label='MSSFA')
ax_batch.set_yscale('log')
ax_batch.set_title(f'Batch Algorithm Runtime ($J={const_J}$)')
ax_batch.set_xlabel('Input Signals')
ax_batch.set_ylabel('Training Time (s)')
ax_batch.legend()

fig_inc, ax_inc = plt.subplots()
ax_inc.plot(x_vals, lag_times[4, :], '--o', label='RSFA')
ax_inc.plot(x_vals, lag_times[5, :], '--s', label='IncSFA')
ax_inc.plot(x_vals, lag_times[6, :], '--^', label='IncSFA SVD')
ax_inc.plot(x_vals, lag_times[7, :], '--P', label='IncMSSFA')
ax_inc.set_yscale('log')
ax_inc.set_title(f'Incremental Algorithm Runtime ($J={const_J}$)')
ax_inc.set_xlabel('Input Signals')
ax_inc.set_ylabel('Training Time (s)')
ax_inc.legend()

"""Runtimes for different J values"""
X, T0, T4, T5, T10 = tepimport.import_tep_sets(lagged_samples=const_lag)
m, n = X.shape

X_mean = np.mean(X, axis=1).reshape((-1, 1))
X = X - X_mean
X_std = np.std(X, axis=1).reshape((-1, 1))
X = X / X_std
J_iteration = -1

step_size = floor(m / n_J)

for J in range(step_size, m + 1, step_size):
    print(f'J={J}')
    J_iteration += 1

    for iteration in range(n_iters_J):
        """Import Data"""
        sfa_object = sfa.SFA()
        ssfa_object = ssfa.SSFA()
        spca_object = SparsePCA(n_components=J, max_iter=500, tol=1e-6)
        mssfa_object = mssfa.MSSFA("chol", "l1")
        rsfa_object = rsfa.RSFA(m, J, J, L=2, conv_tol=0)
        incsfa_object = incsfa.IncSFA(m, J, J, L=2, conv_tol=0)
        incsfa_svd_object = incsfa.IncSFA(m, J, J, L=2, conv_tol=0)
        incmssfa_object = incmssfa.IncMSSFA()

        print("    SFA")
        sfa_time = time.time()
        sfa_object.run(X, J)
        sfa_time = time.time() - sfa_time

        print("    SSFA")
        ssfa_time = time.time()
        ssfa_object.run(X, J)
        ssfa_time = time.time() - ssfa_time

        print("    SPCA")
        spca_time = time.time()
        spca_object.fit(X.T)
        spca_time = time.time() - spca_time

        print("    MSSFA")
        mssfa_time = time.time()
        mssfa_object.run(X, J)
        mssfa_time = time.time() - mssfa_time

        print("    RSFA")
        rsfa_time = time.time()
        for i in range(n):
            rsfa_object.add_data(X[:, i])
        rsfa_time = time.time() - rsfa_time

        print("    IncSFA")
        incsfa_time = time.time()
        for i in range(n):
            incsfa_object.add_data(X[:, i], update_monitors=False,
                                   use_svd_whitening=False)
        incsfa_time = time.time() - incsfa_time

        print("    IncSFA SVD")
        incsfa_svd_time = time.time()
        for i in range(n):
            incsfa_svd_object.add_data(X[:, i], update_monitors=False,
                                       use_svd_whitening=True)
        incsfa_svd_time = time.time() - incsfa_svd_time

        print("    IncMSSFA")
        incmssfa_time = time.time()
        incmssfa_object.run(X, J, L=2)
        incmssfa_time = time.time() - incmssfa_time

        J_times[0, J_iteration, iteration] = sfa_time
        J_times[1, J_iteration, iteration] = ssfa_time
        J_times[2, J_iteration, iteration] = spca_time
        J_times[3, J_iteration, iteration] = mssfa_time
        J_times[4, J_iteration, iteration] = rsfa_time
        J_times[5, J_iteration, iteration] = incsfa_time
        J_times[6, J_iteration, iteration] = incsfa_svd_time
        J_times[7, J_iteration, iteration] = incmssfa_time

J_times = J_times.mean(axis=2)
x_vals = list(range(step_size, m + 1, step_size))

fig_batch_J, ax_batch_J = plt.subplots()
ax_batch_J.plot(x_vals, J_times[0, :], '--o', label='SFA')
ax_batch_J.plot(x_vals, J_times[1, :], '--s', label='SSFA')
ax_batch_J.plot(x_vals, J_times[2, :], '--^', label='SPCA')
ax_batch_J.plot(x_vals, J_times[3, :], '--P', label='MSSFA')
ax_batch_J.set_yscale('log')
ax_batch_J.set_title(f'Batch Algorithm Runtime ($m={m}$)')
ax_batch_J.set_xlabel('Extracted Features')
ax_batch_J.set_ylabel('Training Time (s)')
ax_batch_J.legend()

fig_inc_J, ax_inc_J = plt.subplots()
ax_inc_J.plot(x_vals, J_times[4, :], '--o', label='RSFA')
ax_inc_J.plot(x_vals, J_times[5, :], '--s', label='IncSFA')
ax_inc_J.plot(x_vals, J_times[6, :], '--^', label='IncSFA SVD')
ax_inc_J.plot(x_vals, J_times[7, :], '--P', label='IncMSSFA')
ax_inc_J.set_yscale('log')
ax_inc_J.set_title(f'Incremental Algorithm Runtime ($m={m}$)')
ax_inc_J.set_xlabel('Extracted Features')
ax_inc_J.set_ylabel('Training Time (s)')
ax_inc_J.legend()
plt.show()
