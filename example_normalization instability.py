import src.sfamanopt.mssfa as mssfa
import src.sfamanopt.ssfa as ssfa
import src.sfamanopt.fault_diagnosis as fd

import numpy as np
import matplotlib.pyplot as plt

import tepimport

if __name__ == "__main__":
    alpha = 0.01
    Md_ssfa = 74
    Md_mssfa = 85
    lagged_samples = 2
    # Algorithm names for labels
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

    """Train Models"""
    ssfa_object = ssfa.SSFA()
    W_ssfa, _, _, _, _ = ssfa_object.run(X, Md_ssfa)
    W_ssfa_norm, _, _, _, _ = ssfa_object.run(X_norm, Md_ssfa)
    Lambda_inv_ssfa = np.linalg.pinv(W_ssfa.T @ W_ssfa)
    Lambda_inv_ssfa_norm = np.linalg.pinv(W_ssfa_norm.T @ W_ssfa_norm)

    mssfa_object = mssfa.MSSFA("chol", "l1")
    W_mssfa, _, _, _, _ = mssfa_object.run(X, Md_mssfa)
    W_mssfa_norm, _, _, _, _ = mssfa_object.run(X_norm, Md_mssfa)

    """Test data"""
    n_test = T.shape[1]
    X_test = T - X_mean
    X_test_norm = (T - X_mean) / X_std

    Y_ssfa = W_ssfa.T @ X_test
    Y_ssfa_norm = W_ssfa_norm.T @ X_test_norm

    Y_mssfa = W_mssfa.T @ X_test
    Y_mssfa_norm = W_mssfa_norm.T @ X_test_norm

    # Calculate T^2 for the code from the paper
    T_sqr = np.zeros((4, n_test))
    for i in range(n_test):
        T_sqr[0, i] = Y_ssfa[:, i].T @ Lambda_inv_ssfa @ Y_ssfa[:, i]
        T_sqr[1, i] = (Y_ssfa_norm[:, i].T @ Lambda_inv_ssfa_norm
                       @ Y_ssfa_norm[:, i])
        T_sqr[2, i] = Y_mssfa[:, i].T @ Y_mssfa[:, i]
        T_sqr[3, i] = Y_mssfa_norm[:, i].T @ Y_mssfa_norm[:, i]

    """Plot the comparison"""
    _f, axs2d = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row')
    _f.set_size_inches(21, 9)
    fontsize = 24

    mssfa_plot = axs2d[0][0]
    mssfa_plot.set_title(f"Unnormalized Inputs", fontsize=fontsize)
    mssfa_plot.set_ylabel("Sparse SFA $T^2$", fontsize=fontsize)
    mssfa_plot.plot(T_sqr[0, :])

    mssfa_plot_norm = axs2d[0][1]
    mssfa_plot_norm.set_title(f"Normalized Inputs", fontsize=fontsize)
    mssfa_plot_norm.plot(T_sqr[1, :])

    ssfa_plot = axs2d[1][0]
    ssfa_plot.set_ylabel("Manifold Sparse SFA $T^2$", fontsize=fontsize)
    ssfa_plot.set_xlabel("Sample", fontsize=fontsize)
    ssfa_plot.plot(T_sqr[2, :])

    ssfa_plot_norm = axs2d[1][1]
    ssfa_plot_norm.set_xlabel("Sample", fontsize=fontsize)
    ssfa_plot_norm.plot(T_sqr[3, :])

    _f.set_tight_layout(True)
    plt.savefig(f"plots/normalized_comparison.png", dpi=350)
    plt.close(fig=_f)
    _f = None
