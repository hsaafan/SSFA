"""
Looks at a sample range, and plots the variable contributions over that range.
Variables are chosen by taking the maximum value of each variable over that
range and then picking the largest n_to_plot variables.
"""
import src.sfamanopt.ssfa as ssfa
import src.sfamanopt.fault_diagnosis as fd

import numpy as np
import matplotlib.pyplot as plt

import tepimport

if __name__ == "__main__":
    alpha = 0.01
    Md = 90
    lagged_samples = 2
    fd_method = 'CDC'
    fd_samples = [i for i in range(153, 164)]  # Contribution plot samples
    n_to_plot = 10  # Contribution plot number of variables to plot
    idv = [4, 5]
    # Linestyles for the n_to_plot variables
    linestyles = ['--s', '--o', '--<', '-->', '--H',
                  ':s', ':o', ':<', ':>', ':H']
    """Import Data"""
    ignored_var = list(range(22, 41))
    X = tepimport.import_sets(0, skip_test=True)[0][1]
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
    ssfa_object = ssfa.SSFA("chol", "l1")
    W, cost_values, sparsity_values, relative_errors = ssfa_object.run(X, m)

    """Order Features and take slowest subset"""
    Y = W.T @ X
    Y_dot = Y[:, 1:] - Y[:, :-1]
    speeds = np.diag(Y_dot @ Y_dot.T) / n
    order = np.argsort(speeds)
    Omega_inv = np.diag(speeds[order] ** -1)
    W = W[:, order]

    U, Lam, UT = np.linalg.svd(np.cov(X))
    Q = U @ np.diag(Lam ** -(1/2))
    Z = Q.T @ X
    Z_dot = Z[:, 1:] - Z[:, :-1]
    P, Omega, PT = np.linalg.svd(np.cov(Z_dot))
    W_orig = Q @ P
    Omega_inv_orig = np.diag(Omega ** -1)

    results = []
    for name, test_data in tests:
        X_test = ((test_data - X_mean) / X_std)
        X_cont = X_test[:, fd_samples]
        Y = (W.T @ X_test)
        stats = fd.calculate_test_stats(Y, Md, Omega_inv)
        conts = fd.calculate_fault_contributions(X_cont, W.T, Omega_inv,
                                                 Md, fd_method)

        Y_orig = (W_orig.T @ X_test)
        stats_orig = fd.calculate_test_stats(Y_orig, Md, Omega_inv_orig)
        conts_orig = fd.calculate_fault_contributions(X_cont, W_orig.T,
                                                      Omega_inv_orig,
                                                      Md, fd_method)

        results.append((name, stats, conts, stats_orig, conts_orig))

    Tdc, Tec, Sdc, Sec = fd.calculate_crit_values(n_test, Md, Me, alpha)

    index_labels = [
        "XMEAS(1) A Feed  (stream 1) kscmh",
        "XMEAS(2) D Feed  (stream 2) kg/hr",
        "XMEAS(3) E Feed  (stream 3) kg/hr",
        "XMEAS(4) A and C Feed  (stream 4) kscmh",
        "XMEAS(5) Recycle Flow  (stream 8) kscmh",
        "XMEAS(6) Reactor Feed Rate  (stream 6) kscmh",
        "XMEAS(7) Reactor Pressure kPa gauge",
        "XMEAS(8) Reactor Level %",
        "XMEAS(9) Reactor Temperature Deg C",
        "XMEAS(10) Purge Rate (stream 9) kscmh",
        "XMEAS(11) Product Sep Temp Deg C",
        "XMEAS(12) Product Sep Level %",
        "XMEAS(13) Prod Sep Pressure kPa gauge",
        "XMEAS(14) Prod Sep Underflow (stream 10) m3/hr",
        "XMEAS(15) Stripper Level %",
        "XMEAS(16) Stripper Pressure kPa gauge",
        "XMEAS(17) Stripper Underflow (stream 11) m3/hr",
        "XMEAS(18) Stripper Temperature Deg C",
        "XMEAS(19) Stripper Steam Flow kg/hr",
        "XMEAS(20) Compressor Work kW",
        "XMEAS(21) Reactor Cooling Water Outlet Temp Deg C",
        "XMEAS(22) Separator Cooling Water Outlet Temp Deg C",
        "XMV(1) D Feed Flow (stream 2)",
        "XMV(2) E Feed Flow (stream 3)",
        "XMV(3) A Feed Flow (stream 1)",
        "XMV(4) A and C Feed Flow (stream 4)",
        "XMV(5) Compressor Recycle Valve",
        "XMV(6) Purge Valve (stream 9)",
        "XMV(7) Separator Pot Liquid Flow (stream 10)",
        "XMV(8) Stripper Liquid Product Flow (stream 11)",
        "XMV(9) Stripper Steam Valve",
        "XMV(10) Reactor Cooling Water Flow",
        "XMV(11) Condenser Cooling Water Flow",
    ]
    lagged_labels = []
    for i in range(1, lagged_samples + 1):
        for lbl in index_labels:
            lagged_labels.append(f"{lbl} (t - {i})")
    index_labels += lagged_labels

    for name, stats, conts, stats_orig, conts_orig in results:
        """Plot Stats"""
        _f, axs2d = plt.subplots(nrows=4, ncols=1, sharex=True)
        _f.set_size_inches(21, 9)

        Td_plot = axs2d[0]
        Td_plot.set_title(f"{name} Test Statistics")
        Td_plot.set_ylabel("$T^2_d$")
        Td_plot.plot(stats[0], label='Manifold Sparse SFA')
        Td_plot.plot(stats_orig[0], label='SFA')
        Td_plot.plot([Tdc] * len(stats[0]))
        Td_plot.legend(loc='upper left')

        Te_plot = axs2d[1]
        Te_plot.set_ylabel("$T^2_e$")
        Te_plot.plot(stats[1], label='Manifold Sparse SFA')
        Te_plot.plot(stats_orig[1], label='SFA')
        Te_plot.plot([Tec] * len(stats[1]))
        Te_plot.legend(loc='upper left')

        Sd_plot = axs2d[2]
        Sd_plot.set_ylabel("$S^2_d$")
        Sd_plot.plot(stats[2], label='Manifold Sparse SFA')
        Sd_plot.plot(stats_orig[2], label='SFA')
        Sd_plot.plot([Sdc] * len(stats[2]))
        Sd_plot.legend(loc='upper left')

        Se_plot = axs2d[3]
        Se_plot.set_ylabel("$S^2_e$")
        Se_plot.plot(stats[3], label='Manifold Sparse SFA')
        Se_plot.plot(stats_orig[3], label='SFA')
        Se_plot.plot([Sec] * len(stats[3]))
        Se_plot.legend(loc='upper left')
        Se_plot.set_xlabel("Sample")

        plt.savefig(f'plots/CS/{name}_stats.png', dpi=350)
        plt.close(fig=_f)
        _f = None

        Td_cont, Te_cont, Sd_cont, Se_cont = conts
        Td_cont_orig, Te_cont_orig, Sd_cont_orig, Se_cont_orig = conts_orig
        for stat_name, cont in [('Td', Td_cont), ('Te', Te_cont),
                                ('Sd', Sd_cont), ('Se', Se_cont),
                                ('Td_orig', Td_cont_orig),
                                ('Te_orig', Te_cont_orig),
                                ('Sd_orig', Sd_cont_orig),
                                ('Se_orig', Se_cont_orig)]:
            largest_cont_over_range = np.max(cont, axis=1)
            order = np.argsort(-1 * largest_cont_over_range)
            _f, ax = plt.subplots()
            _f.set_size_inches(16, 9)
            if stat_name[0] == "S":
                indices = [i + lagged_samples + 1 for i in fd_samples[:-1]]
            else:
                indices = [i + lagged_samples for i in fd_samples]
            for i in range(min(n_to_plot, len(order))):
                index = order[i]
                ax.plot(indices,
                        cont[index, :],
                        linestyles[i],
                        label=index_labels[index])
            ax.legend(loc='upper left')
            ax.set_xlabel('Sample')
            ax.set_xticks(indices)
            ax.set_ylabel('Contribution')
            title = (f'{name}: Top {n_to_plot} Contributing Variables to '
                     f'${stat_name[0]}_{stat_name[1]}^2$ For Samples '
                     f'{indices[0]} - {indices[-1]}')
            ax.set_title(title)
            plt.savefig(f'plots/CS/{name}_{stat_name}.png', dpi=350)
            plt.close(fig=_f)
