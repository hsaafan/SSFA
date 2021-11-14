import src.sfamanopt.ssfa as ssfa
import src.sfamanopt.paper_ssfa as oldssfa
import src.sfamanopt.fault_diagnosis as fd

import numpy as np
import matplotlib.pyplot as plt

import tepimport

if __name__ == "__main__":
    alpha = 0.01
    Md = 90
    lagged_samples = 2
    # Algorithm names for labels
    us = "Manifold Sparse SFA"
    them = "Sparse SFA"
    """Import Data"""
    X = tepimport.import_sets((0), skip_test=True)[0]
    T = tepimport.import_sets((4), skip_training=True)[0]

    ignored_var = list(range(22, 41))
    X = np.delete(X[1], ignored_var, axis=0)
    T = np.delete(T[1], ignored_var, axis=0)

    X = tepimport.add_lagged_samples(X, lagged_samples)
    T = tepimport.add_lagged_samples(T, lagged_samples)

    m = X.shape[0]
    n = X.shape[1]
    X_mean = np.mean(X, axis=1).reshape((-1, 1))
    X = X - X_mean
    X_std = np.std(X, axis=1).reshape((-1, 1))
    X_norm = X / X_std
    Me = m - Md

    """Train Models"""
    ssfa_object = ssfa.SSFA("chol", "l1")
    paper_ssfa_object = oldssfa.PaperSSFA()
    W, _, _, _ = ssfa_object.run(X, Md)
    W_norm, _, _, _ = ssfa_object.run(X_norm, Md)
    W_old, _, _, _ = paper_ssfa_object.run(X, Md, mu=5)
    W_old_norm, _, _, _ = paper_ssfa_object.run(X_norm, Md, mu=5)

    # Used to normalize Td in the monitoring statistic calculation
    Lambda_inv_old = np.linalg.pinv(W_old.T @ W_old)
    Lambda_inv_old_norm = np.linalg.pinv(W_old_norm.T @ W_old_norm)

    """Test data"""
    n_test = T.shape[1]
    X_test = T - X_mean
    X_test_norm = (T - X_mean) / X_std

    Y = W.T @ X_test
    Y_norm = W_norm.T @ X_test_norm

    Y_old = W_old.T @ X_test
    Y_old_norm = W_old_norm.T @ X_test_norm

    # Calculate T^2 for the code from the paper
    T_sqr = np.zeros((4, n_test))
    for i in range(n_test):
        T_sqr[0, i] = Y[:, i].T @ Y[:, i]
        T_sqr[1, i] = Y_norm[:, i].T @ Y_norm[:, i]
        T_sqr[2, i] = Y_old[:, i].T @ Lambda_inv_old @ Y_old[:, i]
        T_sqr[3, i] = (Y_old_norm[:, i].T @ Lambda_inv_old_norm
                       @ Y_old_norm[:, i])

    Tdc, Tec, Sdc, Sec = fd.calculate_crit_values(n_test, Md, Me, alpha)

    """Plot the comparison"""
    _f, axs2d = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row')
    _f.set_size_inches(21, 9)
    fontsize = 24

    us = axs2d[0][0]
    us.set_title(f"Unnormalized Inputs", fontsize=fontsize)
    us.set_ylabel("Manifold Sparse SFA $T^2$", fontsize=fontsize)
    us.plot(T_sqr[0, :])

    us_norm = axs2d[0][1]
    us_norm.set_title(f"Normalized Inputs", fontsize=fontsize)
    us_norm.plot(T_sqr[1, :])

    them = axs2d[1][0]
    them.set_ylabel("Sparse SFA $T^2$", fontsize=fontsize)
    them.set_xlabel("Sample", fontsize=fontsize)
    them.plot(T_sqr[2, :])

    them_norm = axs2d[1][1]
    them_norm.set_xlabel("Sample", fontsize=fontsize)
    them_norm.plot(T_sqr[3, :])

    _f.set_tight_layout(True)
    plt.savefig(f"Normalized_comparison.png", dpi=350)
    plt.close(fig=_f)
    _f = None
