""" Choose the lag number and the number of SF/PC kept """
from src.sfamanopt import incmssfa
from SlowFeatureAnalysis.src.sfafd import incsfa, rsfa
import src.sfamanopt.load_cva as cva
import src.sfamanopt.fault_diagnosis as fd

import numpy as np
import scipy

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
                      max_lag: int, speed_kept: float, alpha: float,
                      skip_rsfa: bool = False, skip_incsfa: bool = False,
                      skip_incmssfa: bool = False, verbose: bool = True,
                      continuous_print: bool = False):
    results_rsfa = []
    results_incsfa = []
    results_incmssfa = []

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
        n_test = T.shape[1]
        while fault_indices[-1] >= n_test:
            fault_indices = fault_indices[:-1]

        upper_speed = upper_quantile_x_speed(X, 1 - speed_kept)

        """Train Models"""
        if not skip_rsfa:
            message('    Testing RSFA', verbose)
            rsfa_object = rsfa.RSFA(m, m, m, 2, conv_tol=0)
            Y = np.zeros((m, n))
            for i in range(n):
                y, _, _ = rsfa_object.add_data(X[:, i])
                Y[:, i] = y.flat
            W = (rsfa_object.standardization_node.whitening_matrix
                 @ rsfa_object.transformation_matrix)
            Y = W.T @ X
            Md = calculate_Md(Y, n, upper_speed)
            Me = m - Md

            T2c = fd.calculate_crit_values(n, Md, Me, alpha)[0]
            W = W[:, :Md]
            Y_V = (W.T @ V)
            Y_T = (W.T @ T)

            metrics = calculate_FAR_FDR(Y_V, Y_T, T2c, fault_indices)
            results_rsfa.append([Md, metrics[0], metrics[1], metrics[2]])
            if continuous_print:
                print(results_rsfa[-1])

        if not skip_incsfa:
            message('    Testing IncSFA', verbose)
            incsfa_object = incsfa.IncSFA(m, m, m, 2, conv_tol=0)
            Y = np.zeros((m, n))
            for i in range(n):
                y, _, _ = incsfa_object.add_data(X[:, i],
                                                 use_svd_whitening=True)
                Y[:, i] = y.flat
            W = (incsfa_object.standardization_node.whitening_matrix
                 @ incsfa_object.transformation_matrix)
            Y = W.T @ X
            Md = calculate_Md(Y, n, upper_speed)
            Me = m - Md

            T2c = fd.calculate_crit_values(n, Md, Me, alpha)[0]
            W = W[:, :Md]
            Y_V = (W.T @ V)
            Y_T = (W.T @ T)

            metrics = calculate_FAR_FDR(Y_V, Y_T, T2c, fault_indices)
            results_incsfa.append([Md, metrics[0], metrics[1], metrics[2]])
            if continuous_print:
                print(results_incsfa[-1])

        if not skip_incmssfa:
            message('    Testing IncMSSFA', verbose)
            incmssfa_object = incmssfa.IncMSSFA()
            W, _, _ = incmssfa_object.run(X, m, L=2)
            Y = (W.T @ X)
            Md = calculate_Md(Y, n, upper_speed)
            Me = m - Md

            T2c = fd.calculate_crit_values(n, Md, Me, alpha)[0]
            W, _, _ = incmssfa_object.run(X, Md, L=2)
            Y_V = (W.T @ V)
            Y_T = (W.T @ T)

            metrics = calculate_FAR_FDR(Y_V, Y_T, T2c, fault_indices)
            results_incmssfa.append([Md, metrics[0], metrics[1], metrics[2]])
            if continuous_print:
                print(results_incmssfa[-1])

        if np.all(np.asarray([skip_rsfa, skip_incsfa, skip_incmssfa])):
            break
        fault_indices = [i - 1 for i in fault_indices]

    return(results_rsfa, results_incsfa, results_incmssfa)


if __name__ == '__main__':
    speed_kept = 0.9
    alpha = 0.01
    lag_range = [2, 2]   # [Min, Max] lags to test
    dataset = 'CVA'      # ['TEP', 'CVA']
    skip_rsfa = False
    skip_incsfa = False
    skip_incmssfa = False

    """Import Data"""
    if dataset == 'TEP':
        X, T0, _, _, _ = tepimport.import_tep_sets(lagged_samples=0)
        T1 = tepimport.import_sets(1, skip_training=True)[0][1]  # IDV 1
        T1 = np.delete(T1, list(range(22, 41)), axis=0)
        fault_indices = np.asarray([i for i in range(160, 960)])
    elif dataset == 'CVA':
        data_sets_unlagged = cva.import_sets(lagged_samples=0)
        X = np.hstack([data_sets_unlagged[0][1], data_sets_unlagged[2][1]])
        T0 = data_sets_unlagged[1][1]
        T1 = data_sets_unlagged[9][1]  # Fault 3.1
        X = X[:-1, :]    # Remove var 24
        T0 = T0[:-1, :]  # Remove var 24
        T1 = T1[:-1, :]  # Remove var 24
        fault_range = data_sets_unlagged[9][2]
        fault_indices = np.asarray([i for i in range(*fault_range)])

    print(choose_parameters(X, T0, T1, fault_indices, lag_range[0],
                            lag_range[1], speed_kept, alpha,
                            skip_rsfa, skip_incsfa, skip_incmssfa,
                            continuous_print=True))
