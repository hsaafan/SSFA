"""
Looks at a sample range, and plots the variable contributions over that range.
Variables are chosen by taking the maximum value of each variable over that
range and then picking the largest n_to_plot variables.
"""
import src.sfamanopt.mssfa as mssfa
import src.sfamanopt.ssfa as ssfa
import src.sfamanopt.sfa as sfa
import src.sfamanopt.load_cva as cva

import numpy as np
import scipy
from sklearn.decomposition import SparsePCA

import tepimport
from math import floor


def upper_quantile_x_speed(X: np.ndarray, q: float) -> float:
    m, n = X.shape

    X_dot = X[:, 1:] - X[:, :-1]
    speeds = np.sum(X_dot ** 2, axis=1) / (n - 1)
    ordered_speeds = np.sort(speeds)
    speed_index = floor(m - (m * q))
    return(ordered_speeds[speed_index])


if __name__ == "__main__":
    speed_kept = 0.9
    var_kept = 0.9
    lag_range = [2, 2]  # [Min, Max] lags to test
    dataset = 'CVA'  # ['TEP', 'CVA']

    results = []
    for lag in range(lag_range[0], lag_range[1] + 1):
        print(f"Testing lag = {lag}")

        """Import Data"""
        if dataset == 'TEP':
            ignored_var = list(range(22, 41))
            X, T0, _, _, _ = tepimport.import_tep_sets(lagged_samples=lag)
        elif dataset == 'CVA':
            data_sets_unlagged = cva.import_sets(lagged_samples=0)
            X = np.hstack([x[1] for x in data_sets_unlagged[1:3]])
            X = X[:-1, :]  # Remove var 24
            X = cva.add_lagged_samples(X, lagged_samples=lag)

        """Preprocess Data"""
        m = X.shape[0]
        n = X.shape[1]
        X_mean = np.mean(X, axis=1).reshape((-1, 1))
        X = X - X_mean
        X_std = np.std(X, axis=1).reshape((-1, 1))
        X = X / X_std
        upper_speed = upper_quantile_x_speed(X, 1 - speed_kept)

        """Train Models"""
        sfa_object = sfa.SFA()
        W_sfa, Omega_inv_sfa = sfa_object.run(X, m)
        Y_sfa = (W_sfa.T @ X)
        Y_sfa_dot = Y_sfa[:, 1:] - Y_sfa[:, :-1]
        speeds_sfa = np.sum(Y_sfa_dot ** 2, axis=1) / (n - 1)
        Md_sfa = np.argwhere(np.sort(speeds_sfa) > upper_speed)[0]

        ssfa_object = ssfa.SSFA()
        W_ssfa, Omega_inv_ssfa, _, _, _ = ssfa_object.run(X, m)
        Lambda_inv_ssfa = np.linalg.pinv(W_ssfa.T @ W_ssfa)
        Y_ssfa = scipy.linalg.sqrtm(Lambda_inv_ssfa) @ (W_ssfa.T @ X)
        Y_ssfa_dot = Y_ssfa[:, 1:] - Y_ssfa[:, :-1]
        speeds_ssfa = np.sum(Y_ssfa_dot ** 2, axis=1) / (n - 1)
        Md_ssfa = np.argwhere(np.sort(speeds_ssfa) > upper_speed)[0]

        mssfa_object = mssfa.MSSFA("chol", "l1")
        W_mssfa, Omega_inv_mssfa, _, _, _ = mssfa_object.run(X, m)
        Y_mssfa = (W_mssfa.T @ X)
        Y_mssfa_dot = Y_mssfa[:, 1:] - Y_mssfa[:, :-1]
        speeds_mssfa = np.sum(Y_mssfa_dot ** 2, axis=1) / (n - 1)
        Md_mssfa = np.argwhere(np.sort(speeds_mssfa) > upper_speed)[0]

        total_var = np.sum(np.var(X, axis=1))
        K_min = 1
        K_max = m
        while True:
            if K_max - K_min <= 3:
                for i in range(K_min, K_max + 1):
                    print(f"Checking K = {i}")
                    spca = SparsePCA(n_components=i, max_iter=500, tol=1e-6)
                    Y_spca = spca.fit_transform(X.T)
                    if np.sum(np.var(Y_spca, axis=0)) >= var_kept * total_var:
                        K_spca = i
                        break
                break
            K_spca = floor((K_min + K_max) / 2)
            print(f"Checking K = {K_spca}")
            spca = SparsePCA(n_components=K_spca, max_iter=500, tol=1e-6)
            Y_spca = spca.fit_transform(X.T)
            if np.sum(np.var(Y_spca, axis=0)) >= var_kept * total_var:
                K_max = K_spca
            else:
                K_min = K_spca

        results.append([lag, Md_sfa, Md_ssfa, K_spca, Md_mssfa])

    print(results)
