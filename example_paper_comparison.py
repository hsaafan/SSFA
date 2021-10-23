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
    Md = 30
    lagged_samples = 1
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
    W, _, _, _ = ssfa_object.run(X, m)
    W_old, _, _, _ = paper_ssfa_object.run(X, Md, mu=5)

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

    tests = [("IDV(10)", T10), ("IDV(11)", T11)]
    n_test = T10.shape[1]

    results = []
    for name, test_data in tests:
        X_test = ((test_data - X_mean) / X_std)
        Y = (W.T @ X_test)
        Y_old = W_old.T @ X_test

        test_stats = fd.calculate_test_stats(Y, Md, Omega_inv)
        test_stats_old = fd.calculate_test_stats(Y_old, Md, Omega_inv_old)

        results.append((name, *test_stats, *test_stats_old))

    Tdc, Tec, Sdc, Sec = fd.calculate_crit_values(n_test, Md, Me, alpha)

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
