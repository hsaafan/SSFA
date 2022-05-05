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
    """Find the speed at which to split features into slow and fast"""
    m, n = X.shape

    X_dot = X[:, 1:] - X[:, :-1]
    speeds = np.sum(X_dot ** 2, axis=1) / (n - 1)
    ordered_speeds = np.sort(speeds)
    speed_index = floor(m - (m * q))
    return(ordered_speeds[speed_index])


def calculate_Md(Y: np.ndarray, n: int, upper_speed: float) -> int:
    """Calculates the number of features to keep for SFA based algorithms"""
    Y_dot = Y[:, 1:] - Y[:, :-1]
    speeds = np.sum(Y_dot ** 2, axis=1) / (n - 1)
    return(int(np.argwhere(np.sort(speeds) > upper_speed)[0]))


def message(msg: str, print_msg: bool) -> None:
    """Helper function for cleaner looking code"""
    if print_msg:
        print(msg)


def calculate_T2(Y: np.ndarray, gamma_inv: np.ndarray = None) -> np.ndarray:
    """Calculate the T^2 statistic"""
    n = Y.shape[1]
    T2 = np.zeros((n, ))
    if gamma_inv is None:
        for i in range(n):
            T2[i] = Y[:, i].T @ Y[:, i]
    else:
        Y = Y.T
        for i in range(n):
            T2[i] = Y[i, :] @ gamma_inv @ Y[i, :].T
    return(T2)


def calculate_FAR_FDR(Y_V: np.ndarray, Y_T: np.ndarray,
                      T2c: float, fault_indices: list,
                      gamma_inv: np.ndarray = None) -> tuple:
    """Calculate FAR and FDR values for a validation and test set"""
    n_validation = Y_V.shape[1]
    n_faults = len(fault_indices)
    n_normal = Y_T.shape[1] - n_faults

    validation_faults = np.sum(calculate_T2(Y_V, gamma_inv) > T2c)
    marked_faults = calculate_T2(Y_T, gamma_inv) > T2c
    true_faults = np.sum(marked_faults[fault_indices])
    false_alarms = np.sum(np.delete(marked_faults, fault_indices))

    FAR_V = validation_faults / n_validation
    FDR_T = true_faults / n_faults
    FAR_T = false_alarms / n_normal
    return(FAR_V, FDR_T, FAR_T)


def choose_parameters(training_set: np.ndarray, validation_set: np.ndarray,
                      test_set: np.ndarray, fault_indices: list, min_lag: int,
                      max_lag: int, speed_kept: float, variance_kept: float,
                      alpha: float, skip_sfa: bool = False,
                      skip_ssfa: bool = False, skip_mssfa: bool = False,
                      skip_spca: bool = False, verbose: bool = True):
    results_sfa = []
    results_ssfa = []
    results_mssfa = []
    results_spca = []

    for lag in range(min_lag, max_lag + 1):
        message(f"Testing lag = {lag}", verbose)
        X = tepimport.add_lagged_samples(training_set, lag)
        V = tepimport.add_lagged_samples(validation_set, lag)
        T = tepimport.add_lagged_samples(test_set, lag)

        """Preprocess Data"""
        m = X.shape[0]
        n = X.shape[1]
        X_mean = np.mean(X, axis=1).reshape((-1, 1))
        X = X - X_mean
        X_std = np.std(X, axis=1).reshape((-1, 1))
        X = X / X_std
        V = (V - X_mean) / X_std
        T = (T - X_mean) / X_std

        upper_speed = upper_quantile_x_speed(X, 1 - speed_kept)

        """Train Models"""
        if not skip_sfa:
            message('    Testing SFA', verbose)
            sfa_object = sfa.SFA()
            W, _ = sfa_object.run(X, m)
            Y = (W.T @ X)
            Md = calculate_Md(Y, n, upper_speed)
            Me = m - Md

            T2c = fd.calculate_crit_values(n, Md, Me, alpha)[0]
            W, _ = sfa_object.run(X, Md)
            Y_V = (W.T @ V)
            Y_T = (W.T @ T)

            metrics = calculate_FAR_FDR(Y_V, Y_T, T2c, fault_indices)
            results_sfa.append([Md, metrics[0], metrics[1], metrics[2]])

        if not skip_ssfa:
            message('    Testing SSFA', verbose)
            ssfa_object = ssfa.SSFA()
            W, _, _, _, _ = ssfa_object.run(X, m)
            correction = scipy.linalg.sqrtm(np.linalg.pinv(W.T @ W))
            Y = correction @ (W.T @ X)
            Md = calculate_Md(Y, n, upper_speed)
            Me = m - Md

            T2c = fd.calculate_crit_values(n, Md, Me, alpha)[0]
            W, _, _, _, _ = ssfa_object.run(X, Md)
            correction = scipy.linalg.sqrtm(np.linalg.pinv(W.T @ W))
            Y_V = correction @ (W.T @ V)
            Y_T = correction @ (W.T @ T)

            metrics = calculate_FAR_FDR(Y_V, Y_T, T2c, fault_indices)
            results_ssfa.append([Md, metrics[0], metrics[1], metrics[2]])

        if not skip_mssfa:
            message('    Testing MSSFA', verbose)
            mssfa_object = mssfa.MSSFA("chol", "l1")
            try:
                W, _, _, _, _ = mssfa_object.run(X, m)
                Y = (W.T @ X)
                Md = calculate_Md(Y, n, upper_speed)
                Me = m - Md

                T2c = fd.calculate_crit_values(n, Md, Me, alpha)[0]
                W, _, _, _, _ = mssfa_object.run(X, Md)
                Y_V = (W.T @ V)
                Y_T = (W.T @ T)

                metrics = calculate_FAR_FDR(Y_V, Y_T, T2c, fault_indices)
                results_mssfa.append([Md, metrics[0], metrics[1], metrics[2]])
            except np.linalg.LinAlgError as e:
                if str(e) == "Matrix is not positive definite":
                    # The covariance matrix is not positive definite meaning
                    # the cholesky retraction can't be used
                    skip_mssfa = True
                else:
                    # For other linear algebra errors, reraise error
                    raise np.linalg.LinAlgError(str(e))

        if not skip_spca:
            message('    Testing SPCA', verbose)
            total_var = np.sum(np.var(X, axis=1))
            var_threshold = total_var * variance_kept
            K_min = 1
            K_max = m
            while True:
                # Search for correct K value
                if K_max - K_min <= 3:
                    for i in range(K_min, K_max + 1):
                        spca = SparsePCA(n_components=i, max_iter=500,
                                         tol=1e-6)
                        Y = spca.fit_transform(X.T)

                        if np.sum(np.var(Y, axis=0)) >= var_threshold:
                            K = i
                            break
                    break
                K = floor((K_min + K_max) / 2)
                spca = SparsePCA(n_components=K, max_iter=500, tol=1e-6)
                Y = spca.fit_transform(X.T)
                if np.sum(np.var(Y, axis=0)) >= var_threshold:
                    K_max = K
                else:
                    K_min = K

            P = spca.components_.T[:, :K]
            T2c = fd.calculate_crit_values_pca(X.T, P, n, K, m - K, alpha)[0]

            P_d = P[:, :K]
            gamma_inv = np.linalg.inv(np.cov((X.T @ P_d).T))

            scores_V = (V.T @ P_d).T
            scores_T = (T.T @ P_d).T

            metrics = calculate_FAR_FDR(scores_V, scores_T, T2c, fault_indices,
                                        gamma_inv)
            results_spca.append([K, metrics[0], metrics[1], metrics[2]])

        if np.all(np.asarray([skip_sfa, skip_ssfa, skip_mssfa, skip_spca])):
            break
        fault_indices = [i - 1 for i in fault_indices]

    return(results_sfa, results_ssfa, results_mssfa, results_spca)


if __name__ == '__main__':
    speed_kept = 0.9
    var_kept = 0.9
    alpha = 0.01
    lag_range = [0, 11]  # [Min, Max] lags to test
    dataset = 'TEP'      # ['TEP', 'CVA']
    skip_sfa = False
    skip_ssfa = False
    skip_mssfa = False
    skip_spca = False

    """Import Data"""
    if dataset == 'TEP':
        X, T0, _, _, _ = tepimport.import_tep_sets(lagged_samples=0)
        T1 = tepimport.import_sets(1, skip_training=True)[0][1]
        T1 = np.delete(T1, list(range(22, 41)), axis=0)
        fault_indices = np.asarray([i for i in range(160, 960)])
    elif dataset == 'CVA':
        data_sets_unlagged = cva.import_sets(lagged_samples=0)
        X = np.hstack([x[1] for x in data_sets_unlagged[1:3]])
        T0 = data_sets_unlagged[0][1]
        X = X[:-1, :]    # Remove var 24
        T0 = T0[:-1, :]  # Remove var 24

    print(choose_parameters(X, T0, T1, fault_indices, lag_range[0],
                            lag_range[1], speed_kept, var_kept, alpha,
                            skip_sfa, skip_ssfa, skip_mssfa, skip_spca))
