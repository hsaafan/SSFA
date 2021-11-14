import src.sfamanopt.ssfa as ssfa
import src.sfamanopt.fault_diagnosis as fd

import numpy as np
import matplotlib.pyplot as plt

import tepimport

if __name__ == "__main__":
    alpha = 0.000015
    Md = 90
    lagged_samples = 2
    use_original_sfa = True
    idv = range(22)
    """Import Data"""
    ignored_var = list(range(22, 41))
    X = tepimport.import_sets([0], skip_test=True)[0][1]
    X = tepimport.add_lagged_samples(np.delete(X, ignored_var, axis=0),
                                     lagged_samples)

    test_sets = tepimport.import_sets(idv, skip_training=True)
    tests = []
    for name, data in test_sets:
        data = np.delete(data, ignored_var, axis=0)
        data = tepimport.add_lagged_samples(data, lagged_samples)
        tests.append((name, data))
    n_test = tests[0][1].shape[1]

    """Preprocess Data"""
    m = X.shape[0]
    n = X.shape[1]
    X_mean = np.mean(X, axis=1).reshape((-1, 1))
    X = X - X_mean
    X_std = np.std(X, axis=1).reshape((-1, 1))
    X_std = np.ones_like(X_mean)
    X = X / X_std
    Me = m - Md

    """Train Model"""
    ssfa_object = ssfa.SSFA("chol", "l1")
    W, cost_values, sparsity_values, relative_errors = ssfa_object.run(X, m)

    """Order Features and take slowest subset"""
    Y = W.T @ X
    Y_dot = Y[:, 1:] - Y[:, :-1]
    speeds = np.diag(Y_dot @ Y_dot.T) / n
    order = np.argsort(speeds)
    Omega_inv = np.diag(speeds[order] ** -1)
    W = W[:, order]

    if use_original_sfa:
        U, Lam, UT = np.linalg.svd(np.cov(X))
        Q = U @ np.diag(Lam ** -(1/2))
        Z = Q.T @ X
        Z_dot = Z[:, 1:] - Z[:, :-1]
        P, Omega, PT = np.linalg.svd(np.cov(Z_dot))
        W = Q @ P
        Omega_inv = np.diag(Omega ** -1)

    results = []
    for name, test_data in tests:
        X_test = ((test_data - X_mean) / X_std)
        Y = (W.T @ X_test)
        test_stats = fd.calculate_test_stats(Y, Md, Omega_inv)
        results.append((name, *test_stats))

    Tdc, Tec, Sdc, Sec = fd.calculate_crit_values(n_test, Md, Me, alpha)

    for name, Td, Te, Sd, Se in results:
        T_fault_index = 160 - lagged_samples
        S_fault_index = 159 - lagged_samples
        FDR = np.count_nonzero(Td[T_fault_index:] > Tdc) / (len(Td) - T_fault_index)
        FAR = np.count_nonzero(Td[:T_fault_index] > Tdc) / T_fault_index
        if name == "IDV(0)":
            FDR = 0
            FAR = np.count_nonzero(Td > Tdc) / len(Td)
        print(f"{name} FDR = {FDR:.3f} | FAR = {FAR:.3f}")
