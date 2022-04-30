""" Choose the lag number and the number of SF/PC kept """
import src.sfamanopt.mssfa as mssfa
import src.sfamanopt.ssfa as ssfa
import src.sfamanopt.sfa as sfa
import src.sfamanopt.load_cva as cva
import src.sfamanopt.fault_diagnosis as fd

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
    alpha = 0.01
    degenerate_speed = 1e-10  # Below this speed, SF are considered degenrate
    lag_range = [0, 100]  # [Min, Max] lags to test
    dataset = 'TEP'       # ['TEP', 'CVA']

    max_lag = [-1, -1, -1, -1]
    FAR = [[], [], [], []]
    results = []
    for lag in range(lag_range[0], lag_range[1] + 1):
        lag_result = [lag, 0, 0, 0, 0]
        print(f"Testing lag = {lag}")

        """Import Data"""
        if dataset == 'TEP':
            X, T0, _, _, _ = tepimport.import_tep_sets(lagged_samples=lag)
        elif dataset == 'CVA':
            data_sets_unlagged = cva.import_sets(lagged_samples=0)
            X = np.hstack([x[1] for x in data_sets_unlagged[1:3]])
            T0 = data_sets_unlagged[0][1]
            X = X[:-1, :]    # Remove var 24
            T0 = T0[:-1, :]  # Remove var 24
            X = cva.add_lagged_samples(X, lagged_samples=lag)
            T0 = cva.add_lagged_samples(T0, lagged_samples=lag)

        """Preprocess Data"""
        m = X.shape[0]
        n = X.shape[1]
        X_mean = np.mean(X, axis=1).reshape((-1, 1))
        X = X - X_mean
        X_std = np.std(X, axis=1).reshape((-1, 1))
        X = X / X_std
        upper_speed = upper_quantile_x_speed(X, 1 - speed_kept)
        T0 = (T0 - X_mean) / X_std

        """Train Models"""
        if max_lag[0] < 0:
            print('    Testing SFA')
            sfa_object = sfa.SFA()
            W_sfa, Omega_inv_sfa = sfa_object.run(X, m)
            Y_sfa = (W_sfa.T @ X)
            Y_sfa_dot = Y_sfa[:, 1:] - Y_sfa[:, :-1]
            speeds_sfa = np.sum(Y_sfa_dot ** 2, axis=1) / (n - 1)
            Md_sfa = int(np.argwhere(np.sort(speeds_sfa) > upper_speed)[0])
            if np.any(speeds_sfa < degenerate_speed):
                max_lag[0] = lag

            T_sfa = (W_sfa.T @ T0)
            stats_sfa = fd.calculate_test_stats(T_sfa, Md_sfa, Omega_inv_sfa)
            T2c, _, _, _ = fd.calculate_crit_values(n, Md_sfa, m - Md_sfa,
                                                    alpha)
            FAR[0].append(np.sum(stats_sfa[0] > T2c) / T0.shape[1])
            lag_result[1] = Md_sfa

        if max_lag[1] < 0:
            print('    Testing SSFA')
            ssfa_object = ssfa.SSFA()
            W_ssfa, Omega_inv_ssfa, _, _, _ = ssfa_object.run(X, m)
            Lambda_inv_ssfa = np.linalg.pinv(W_ssfa.T @ W_ssfa)
            Y_ssfa = scipy.linalg.sqrtm(Lambda_inv_ssfa) @ (W_ssfa.T @ X)
            Y_ssfa_dot = Y_ssfa[:, 1:] - Y_ssfa[:, :-1]
            speeds_ssfa = np.sum(Y_ssfa_dot ** 2, axis=1) / (n - 1)
            Md_ssfa = int(np.argwhere(np.sort(speeds_ssfa) > upper_speed)[0])
            if np.any(speeds_ssfa < degenerate_speed):
                max_lag[1] = lag

            T_ssfa = (W_ssfa.T @ T0)
            stats_ssfa = fd.calculate_test_stats(T_ssfa, Md_ssfa,
                                                 Omega_inv_ssfa)
            T2c, _, _, _ = fd.calculate_crit_values(n, Md_ssfa, m - Md_ssfa,
                                                    alpha)
            FAR[1].append(np.sum(stats_ssfa[0] > T2c) / T0.shape[1])
            lag_result[2] = Md_ssfa

        if max_lag[2] < 0:
            print('    Testing MSSFA')
            mssfa_object = mssfa.MSSFA("chol", "l1")
            try:
                W_mssfa, Omega_inv_mssfa, _, _, _ = mssfa_object.run(X, m)
                Y_mssfa = (W_mssfa.T @ X)
                Y_mssfa_dot = Y_mssfa[:, 1:] - Y_mssfa[:, :-1]
                speeds_mssfa = np.sum(Y_mssfa_dot ** 2, axis=1) / (n - 1)
                Md_mssfa = int(np.argwhere(np.sort(speeds_mssfa)
                                           > upper_speed)[0])
                if np.any(speeds_mssfa < degenerate_speed):
                    max_lag[2] = lag

                T_mssfa = (W_mssfa.T @ T0)
                stats_mssfa = fd.calculate_test_stats(T_mssfa, Md_mssfa,
                                                      Omega_inv_mssfa)
                T2c, _, _, _ = fd.calculate_crit_values(n, Md_mssfa,
                                                        m - Md_mssfa, alpha)
                FAR[2].append(np.sum(stats_mssfa[0] > T2c) / T0.shape[1])
                lag_result[3] = Md_mssfa
            except np.linalg.LinAlgError as e:
                if str(e) == "Matrix is not positive definite":
                    # The covariance matrix is not positive definite meaning
                    # the cholesky retraction can't be used
                    max_lag[2] = lag - 1
                else:
                    # For other linear algebra errors, reraise error
                    raise np.linalg.LinAlgError(str(e))

        if max_lag[3] < 0:
            print('    Testing SPCA')
            total_var = np.sum(np.var(X, axis=1))
            var_threshold = total_var * var_kept
            K_min = 1
            K_max = m
            while True:
                if K_max - K_min <= 3:
                    for i in range(K_min, K_max + 1):
                        spca = SparsePCA(n_components=i, max_iter=500,
                                         tol=1e-6)
                        Y_spca = spca.fit_transform(X.T)

                        if np.sum(np.var(Y_spca, axis=0)) >= var_threshold:
                            K_spca = i
                            break
                    break
                K_spca = floor((K_min + K_max) / 2)
                spca = SparsePCA(n_components=K_spca, max_iter=500, tol=1e-6)
                Y_spca = spca.fit_transform(X.T)
                if np.sum(np.var(Y_spca, axis=0)) >= var_threshold:
                    K_max = K_spca
                else:
                    K_min = K_spca
            if np.linalg.matrix_rank(spca.components_) < m:
                max_lag[3] = lag

            P = spca.components_.T
            P_d = P[:, :K_spca]
            P_e = P[:, K_spca:]
            scores_d = X.T @ P_d
            scores_e = X.T @ P_e
            gamma_inv_d = np.linalg.inv(np.cov(scores_d.T))
            gamma_inv_e = np.linalg.inv(np.cov(scores_e.T))
            stats_spca = fd.calculate_test_stats_pca(T0.T, P, gamma_inv_d,
                                                     gamma_inv_e, K_spca)
            T2c, _, _, _ = fd.calculate_crit_values_pca(X.T, P, n, K_spca,
                                                        m - K_spca, alpha)
            FAR[3].append(np.sum(stats_spca[0] > T2c) / T0.shape[1])
            lag_result[4] = Md_sfa

        results.append(lag_result)
        if np.all(np.asarray(max_lag) > -1):
            break

    print(f'Max lags: {max_lag}')
    print(FAR)
    print(results)
