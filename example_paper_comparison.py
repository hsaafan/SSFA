import src.sfamanopt.ssfa as ssfa
import src.sfamanopt.paper_ssfa as oldssfa
import src.sfamanopt.fault_diagnosis as fd

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

import tepimport
import pypcfd.plotting as fdplt

if __name__ == "__main__":
    alpha = 0.01
    Md = 25
    lagged_samples = 0
    """Import Data"""
    X = tepimport.import_sets((0), skip_test=True)[0]
    T10, T11 = tepimport.import_sets((10, 11), skip_training=True)

    ignored_var = list(range(22, 41))
    X = np.delete(X[1], ignored_var, axis=0)
    T10 = np.delete(T10[1], ignored_var, axis=0)
    T11 = np.delete(T11[1], ignored_var, axis=0)

    X = tepimport.add_lagged_samples(X, lagged_samples)
    T10 = tepimport.add_lagged_samples(T10, lagged_samples)
    T11 = tepimport.add_lagged_samples(T11, lagged_samples)

    m = X.shape[0]
    n = X.shape[1]
    X_mean = np.mean(X, axis=1).reshape((-1, 1))
    X = X - X_mean
    X_std = np.std(X, axis=1).reshape((-1, 1))
    X = X / X_std
    Me = m - Md

    """Train Models"""
    ssfa_object = ssfa.SSFA("chol", "l1")
    paper_ssfa_object = oldssfa.PaperSSFA()
    W, costs, sparsity, errors = ssfa_object.run(X, m)
    results_from_paper = paper_ssfa_object.run(X, m, mu=5)
    W_old, costs_old, sparsity_old, errors_old = results_from_paper

    plt.subplot(3, 1, 1)
    plt.title("Sparsity")
    plt.plot(sparsity, label='ours')
    plt.plot(sparsity_old, label='paper')
    plt.xlabel("Iteration")
    plt.ylabel("Sparsity")
    plt.legend(loc='upper right')

    plt.subplot(3, 1, 2)
    plt.title("Costs")
    plt.plot(np.asarray(costs) - np.min(costs), label='ours')
    plt.plot(np.asarray(costs_old) - np.min(costs_old), label='paper')
    plt.xlabel("Iteration")
    plt.ylabel("Cost (Shifted to 0 at minimum)")
    plt.legend(loc='upper right')

    plt.subplot(3, 1, 3)
    plt.title("Relative Error")
    plt.plot(errors[2:], label='ours')
    plt.plot(errors_old[2:], label='paper')
    plt.xlabel("Iteration")
    plt.ylabel("Relative Error")
    plt.legend(loc='upper right')
    plt.show()

    """Order Features and take slowest subset"""
    Y = W.T @ X
    Y_dot = Y[:, 1:] - Y[:, :-1]
    speeds = np.diag(Y_dot @ Y_dot.T) / n
    order = np.argsort(speeds)
    Omega_inv = np.diag(speeds[order] ** -1)
    W = W[:, order]

    Y_old = W_old.T @ X
    Y_dot_old = Y_old[:, 1:] - Y_old[:, :-1]
    speeds_old = np.diag(Y_dot_old @ Y_dot_old.T) / n
    order_old = np.argsort(speeds_old)
    Omega_inv_old = np.diag(speeds_old[order_old] ** -1)
    W_old = W_old[:, order_old]

    # Used to normalize Td in the monitoring statistic calculation
    Lambda_inv_old = np.linalg.pinv(W_old.T@W_old)
    # Need to recalculate Y_old, since the W_old in the previous step is
    # already in different order
    Y_old = W_old.T @ X

    tests = [("IDV(10)", T10), ("IDV(11)", T11)]
    n_test = T10.shape[1]

    results = []
    for name, test_data in tests:
        X_test = ((test_data - X_mean) / X_std)
        Y = (W.T @ X_test)
        Y_old = W_old.T @ X_test

        test_stats = fd.calculate_test_stats(Y, Md, Omega_inv)
        test_stats_old = fd.calculate_test_stats(Y_old, Md, Omega_inv_old)
        # Recalculate Td for the code from the paper
        for i in range(n_test):
            test_stats_old[0][i] = Y_old[:Md, i].T @ Lambda_inv_old[:Md, :Md] @ Y_old[:Md, i]
            test_stats_old[1][i] = Y_old[Md:, i].T @ Lambda_inv_old[Md:, Md:] @ Y_old[Md:, i]

        results.append((name, *test_stats, *test_stats_old))

    Tdc, Tec, Sdc, Sec = fd.calculate_crit_values(n_test, Md, Me, alpha)

    _f_sparse, axim = plt.subplots(nrows=1, ncols=2, sharey=True)
    _f_sparse.set_size_inches(8, 6)
    W_old_sparse = np.zeros_like(W_old)
    W_sparse = np.zeros_like(W)
    x_vars = np.diag(np.cov(X))
    for i in range(W_sparse.shape[0]):
        for j in range(W_sparse.shape[1]):
            W_sparse[i, j] = x_vars[i] * W[i, j] / (np.linalg.norm(W[:, j]))
            W_old_sparse[i, j] = x_vars[i] * W_old[i, j] / (np.linalg.norm(W_old[:, j]))

    axim[0].set_title("Sparsity of SSFA")
    axim[1].set_title("Sparsity of SSFA-Old")
    axim[0].imshow(np.abs(W))
    im = axim[1].imshow(np.abs(W_old))
    plt.colorbar(im)

    plt.savefig(f"Sparsity_comparison.png", dpi=350)
    plt.close(fig=_f_sparse)
    _f_sparse = None

    for name, Td, Te, Sd, Se, Td_old, Te_old, Sd_old, Se_old in results:
        _f, axs2d = plt.subplots(nrows=4, ncols=1, sharex=True)
        _f.set_size_inches(8, 6)

        Td_plot = axs2d[0]
        Td_plot.set_title(f"{name} SSFA Comparison")
        Td_plot.set_ylabel("$T^2_d$")
        Td_plot.plot(Td, label='Ours')
        Td_plot.plot(Td_old, label='Paper')
        Td_plot.plot([Tdc] * len(Td))
        Td_plot.legend(loc='upper right')

        Te_plot = axs2d[1]
        Te_plot.set_ylabel("$T^2_e$")
        Te_plot.plot(Te, label='Ours')
        Te_plot.plot(Te_old, label='Paper')
        Te_plot.plot([Tec] * len(Te))
        Te_plot.legend(loc='upper right')

        Sd_plot = axs2d[2]
        Sd_plot.set_ylabel("$S^2_d$")
        Sd_plot.plot(Sd, label='Ours')
        Sd_plot.plot(Sd_old, label='Paper')
        Sd_plot.plot([Sdc] * len(Se))
        Sd_plot.legend(loc='upper right')

        bot_plot = axs2d[3]
        bot_plot.set_ylabel("$S^2_e$")
        bot_plot.plot(Se, label='Ours')
        bot_plot.plot(Se_old, label='Paper')
        bot_plot.plot([Sec] * len(Se))
        bot_plot.set_xlabel("Variable Index")
        bot_plot.legend(loc='upper right')

        plt.savefig(f"{name}_comparison.png", dpi=350)
        plt.close(fig=_f)
        _f = None
