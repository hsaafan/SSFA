import src.sfamanopt.ssfa as mssfa
import src.sfamanopt.paper_ssfa as ssfa
import src.sfamanopt.fault_diagnosis as fd

import numpy as np
import matplotlib.pyplot as plt

import tepimport

if __name__ == "__main__":
    alpha = 0.01
    Md = 55
    lagged_samples = 2
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
    X = X / X_std
    Me = m - Md

    """Train Model"""
    mssfa_object = mssfa.SSFA("chol", "l1")
    W_mssfa, _, _, _ = mssfa_object.run(X, Md)

    ssfa_object = ssfa.PaperSSFA()
    W_ssfa, _, _, _ = ssfa_object.run(X, Md)

    """Order Features and take slowest subset"""
    Y_mssfa = W_mssfa.T @ X
    Y_mssfa_dot = Y_mssfa[:, 1:] - Y_mssfa[:, :-1]
    speeds_mssfa = np.diag(Y_mssfa_dot @ Y_mssfa_dot.T) / n
    order_mssfa = np.argsort(speeds_mssfa)
    Omega_inv_mssfa = np.diag(speeds_mssfa[order_mssfa] ** -1)
    W_mssfa = W_mssfa[:, order_mssfa]

    Y_ssfa = W_ssfa.T @ X
    Y_ssfa_dot = Y_ssfa[:, 1:] - Y_ssfa[:, :-1]
    speeds_ssfa = np.diag(Y_ssfa_dot @ Y_ssfa_dot.T) / n
    order_ssfa = np.argsort(speeds_ssfa)
    Omega_inv_ssfa = np.diag(speeds_ssfa[order_ssfa] ** -1)
    W_ssfa = W_ssfa[:, order_ssfa]
    Lambda_inv_ssfa = np.linalg.pinv(W_ssfa.T @ W_ssfa)

    U, Lam, UT = np.linalg.svd(np.cov(X))
    Q = U @ np.diag(Lam ** -(1/2))
    Z = Q.T @ X
    Z_dot = Z[:, 1:] - Z[:, :-1]
    P, Omega, PT = np.linalg.svd(np.cov(Z_dot))
    W_sfa = Q @ P
    Omega_inv_sfa = np.diag(Omega ** -1)

    results = []
    for name, test_data in tests:
        X_test = ((test_data - X_mean) / X_std)
        Y_sfa = (W_sfa.T @ X_test)
        Y_ssfa = (W_ssfa.T @ X_test)
        Y_mssfa = (W_mssfa.T @ X_test)

        stats_sfa = fd.calculate_test_stats(Y_sfa, Md, Omega_inv_sfa)
        stats_ssfa = fd.calculate_test_stats(Y_ssfa, Md, Omega_inv_ssfa)
        stats_mssfa = fd.calculate_test_stats(Y_mssfa, Md, Omega_inv_mssfa)
        for i in range(n_test):
            stats_ssfa[0][i] = (Y_ssfa[:Md, i].T @ Lambda_inv_ssfa[:Md, :Md]
                                @ Y_ssfa[:Md, i])
            stats_ssfa[1][i] = (Y_ssfa[Md:, i].T @ Lambda_inv_ssfa[Md:, Md:]
                                @ Y_ssfa[Md:, i])

        results.append((name, stats_sfa, stats_ssfa, stats_mssfa))

    Tdc, Tec, Sdc, Sec = fd.calculate_crit_values(n_test, Md, Me, alpha)

    fdr_results = []
    for name, stats_sfa, stats_ssfa, stats_mssfa in results:
        T_fault_index = 160 - lagged_samples
        S_fault_index = 159 - lagged_samples

        idv_results = [name]
        for Td, Te, Sd, Se in [stats_sfa, stats_ssfa, stats_mssfa]:
            if name == "IDV(0)":
                FDR = 0
                FAR = np.count_nonzero(Td > Tdc) / len(Td)
            else:
                FDR = (np.count_nonzero(Td[T_fault_index:] > Tdc)
                       / (len(Td) - T_fault_index))
                FAR = (np.count_nonzero(Td[:T_fault_index] > Tdc)
                       / T_fault_index)
            idv_results.append(FDR)
            idv_results.append(FAR)
        fdr_results.append(idv_results)

    for (name, fdr_sfa, far_sfa,
         fdr_ssfa, far_ssfa,
         fdr_mssfa, far_mssfa) in fdr_results:
        print(f"{name} {fdr_sfa:.3f} {far_sfa:.3f} {fdr_ssfa:.3f} "
              f"{far_ssfa:.3f} {fdr_mssfa:.3f} {far_mssfa:.3f}")
