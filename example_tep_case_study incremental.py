"""
Looks at a sample range, and plots the variable contributions over that range.
Variables are chosen by taking the maximum value of each variable over that
range and then picking the largest n_to_plot variables.
"""
from src.sfamanopt import incmssfa
from SlowFeatureAnalysis.src.sfafd import incsfa, rsfa
import src.sfamanopt.fault_diagnosis as fd

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import tepimport
import time


def plot_test_stats(name: str, idv_name: str,
                    T_stats: list, T_crit: float,
                    S_stats: list, S_crit: float):
    _f, axs2d = plt.subplots(nrows=2, ncols=1, sharex=True)
    _f.set_size_inches(16, 9)

    Td_plot = axs2d[0]
    Td_plot.set_title(f"{name} {idv_name} Test Statistics", fontsize=20)
    Td_plot.set_ylabel("$T^2$", fontsize=20)
    Td_plot.plot(T_stats)
    Td_plot.plot([T_crit] * len(T_stats), 'k')
    Td_plot.set_yscale('log')
    Td_plot.tick_params(axis='both', which='major', labelsize=15)

    Sd_plot = axs2d[1]
    Sd_plot.set_ylabel("$S^2$", fontsize=20)
    Sd_plot.plot(S_stats)
    Sd_plot.plot([S_crit] * len(S_stats), 'k')
    Sd_plot.set_yscale('log')
    Sd_plot.set_xlabel("Sample", fontsize=20)
    Sd_plot.tick_params(axis='both', which='major', labelsize=15)

    _f.set_tight_layout(True)
    return(_f, axs2d)


if __name__ == "__main__":
    alpha = 0.01
    Md = [64, 99, 89, 86]
    lagged_samples = 2
    fd_method = 'CDC'
    fd_samples = [i for i in range(152, 163)]  # Contribution plot samples
    n_to_plot = 5  # Contribution plot number of variables to plot
    idv = [4]
    # Linestyles for the n_to_plot variables
    markers = [(u'#1f77b4', 'o', 'left'),       # XMEAS(01)
               (u'#ff7f0e', 'v', 'full'),       # XMEAS(02)
               (u'#2ca02c', '^', 'full'),       # XMEAS(03)
               (u'#7f7f7f', 'X', 'top'),        # XMEAS(04)
               (u'#9467bd', '>', 'top'),        # XMEAS(05)
               (u'#8c564b', 's', 'left'),       # XMEAS(06)
               (u'#e377c2', 'D', 'full'),       # XMEAS(07)
               (u'#7f7f7f', 'X', 'left'),       # XMEAS(08)
               (u'#bcbd22', 'p', 'full'),       # XMEAS(09)
               (u'#d62728', '<', 'left'),       # XMEAS(10)
               ("black", 'H', 'top'),           # XMEAS(11)
               (u'#ff7f0e', 'v', 'left'),       # XMEAS(12)
               (u'#1f77b4', 'o', 'full'),       # XMEAS(13)
               (u'#2ca02c', '^', 'left'),       # XMEAS(14)
               (u'#17becf', 'P', 'full'),       # XMEAS(15)
               (u'#9467bd', '>', 'left'),       # XMEAS(16)
               (u'#8c564b', 's', 'full'),       # XMEAS(17)
               (u'#e377c2', 'D', 'left'),       # XMEAS(18)
               (u'#7f7f7f', 'X', 'full'),       # XMEAS(19)
               (u'#bcbd22', 'p', 'left'),       # XMEAS(20)
               (u'#17becf', 'P', 'full'),       # XMEAS(21)
               ("black", 'H', 'left'),          # XMEAS(22)
               (u'#1f77b4', 'o', 'top'),        # XMV(01)
               (u'#ff7f0e', 'v', 'top'),        # XMV(02)
               (u'#2ca02c', '^', 'top'),        # XMV(03)
               (u'#d62728', '<', 'top'),        # XMV(04)
               (u'#17becf', 'P', 'top'),        # XMV(05)
               (u'#8c564b', 's', 'top'),        # XMV(06)
               (u'#e377c2', 'D', 'top'),        # XMV(07)
               (u'#bcbd22', 'p', 'top'),        # XMV(08)
               (u'#d62728', '<', 'full'),       # XMV(09)
               (u'#9467bd', '>', 'full'),       # XMV(10)
               ("black", 'H', 'full'),          # XMV(11)
               ]
    linestyles = ['--', '-.', ':']
    styles = []
    for L in linestyles:
        for c, m, f in markers:
            styles.append(dict(marker=m, linestyle=L,
                               fillstyle=f, color=c,
                               alpha=1, markersize=15))
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
    Me = [m - x for x in Md]

    """Train Model"""
    rsfa_object = rsfa.RSFA(m, Md[0], Md[0], 2, conv_tol=0)
    incsfa_object = incsfa.IncSFA(m, Md[1], Md[1], 2, conv_tol=0)
    incsfa_svd_object = incsfa.IncSFA(m, Md[2], Md[2], 2, conv_tol=0)
    incmssfa_object = incmssfa.IncMSSFA()

    Y_rsfa = np.zeros((Md[0], n))
    for i in range(n):
        y, _, _ = rsfa_object.add_data(X[:, i], calculate_monitors=True)
        Y_rsfa[:, i] = y.flat
    rsfa_object.converged = True
    W_rsfa = (rsfa_object.standardization_node.whitening_matrix
              @ rsfa_object.transformation_matrix)
    Omega_inv_rsfa = np.diag(rsfa_object.features_speed ** -1)

    Y_incsfa = np.zeros((Md[1], n))
    for i in range(n):
        y, _, _ = incsfa_object.add_data(X[:, i], use_svd_whitening=False)
        Y_incsfa[:, i] = y.flat
    incsfa_object.converged = True
    W_incsfa = (incsfa_object.standardization_node.whitening_matrix
                @ incsfa_object.transformation_matrix)
    Y_incsfa = W_incsfa.T @ X
    Y_dot_incsfa = Y_incsfa[:, 1:] - Y_incsfa[:, :-1]
    speeds = np.diag(Y_dot_incsfa @ Y_dot_incsfa.T) / n
    Omega_inv_incsfa = np.diag(speeds ** -1)
    Omega_inv_incsfa[np.isinf(Omega_inv_incsfa)] = 0

    Y_incsfa_svd = np.zeros((Md[2], n))
    for i in range(n):
        y, _, _ = incsfa_svd_object.add_data(X[:, i], use_svd_whitening=True)
        Y_incsfa_svd[:, i] = y.flat
    incsfa_svd_object.converged = True
    W_incsfa_svd = (incsfa_svd_object.standardization_node.whitening_matrix
                    @ incsfa_svd_object.transformation_matrix)
    Y_incsfa_svd = W_incsfa_svd.T @ X
    Y_dot_incsfa_svd = Y_incsfa_svd[:, 1:] - Y_incsfa_svd[:, :-1]
    speeds = np.diag(Y_dot_incsfa_svd @ Y_dot_incsfa_svd.T) / n
    Omega_inv_incsfa_svd = np.diag(speeds ** -1)
    Omega_inv_incsfa_svd[np.isinf(Omega_inv_incsfa_svd)] = 0

    W_incmssfa, Omega_inv_incmssfa, _ = incmssfa_object.run(X, Md[3], L=2)

    results = []
    for name, test_data in tests:
        X_test = ((test_data - X_mean) / X_std)
        X_cont = X_test[:, fd_samples]

        Y_rsfa = (W_rsfa.T @ X_test)
        stats_rsfa = fd.calculate_test_stats(Y_rsfa, Md[0], Omega_inv_rsfa)
        conts_rsfa = fd.calculate_fault_contributions(X_cont, W_rsfa.T,
                                                      Omega_inv_rsfa,
                                                      Md[0], fd_method)

        Y_incsfa = (W_incsfa.T @ X_test)
        stats_incsfa = fd.calculate_test_stats(Y_incsfa, Md[1],
                                               Omega_inv_incsfa)
        conts_incsfa = fd.calculate_fault_contributions(X_cont, W_incsfa.T,
                                                        Omega_inv_incsfa,
                                                        Md[1], fd_method)

        Y_incsfa_svd = (W_incsfa_svd.T @ X_test)
        stats_incsfa_svd = fd.calculate_test_stats(Y_incsfa_svd, Md[2],
                                                   Omega_inv_incsfa_svd)
        c = fd.calculate_fault_contributions(X_cont, W_incsfa_svd.T,
                                             Omega_inv_incsfa_svd,
                                             Md[2], fd_method)
        conts_incsfa_svd = c

        Y_incmssfa = (W_incmssfa.T @ X_test)
        stats_incmssfa = fd.calculate_test_stats(Y_incmssfa, Md[3],
                                                 Omega_inv_incmssfa)
        conts_incmssfa = fd.calculate_fault_contributions(X_cont, W_incmssfa.T,
                                                          Omega_inv_incmssfa,
                                                          Md[3], fd_method)

        results.append((name,
                        stats_rsfa, conts_rsfa,
                        stats_incsfa, conts_incsfa,
                        stats_incsfa_svd, conts_incsfa_svd,
                        stats_incmssfa, conts_incmssfa))

    index_labels = [
        "XMEAS(01) A Feed  (stream 1)",
        "XMEAS(02) D Feed  (stream 2)",
        "XMEAS(03) E Feed  (stream 3)",
        "XMEAS(04) A and C Feed  (stream 4)",
        "XMEAS(05) Recycle Flow  (stream 8)",
        "XMEAS(06) Reactor Feed Rate  (stream 6)",
        "XMEAS(07) Reactor Pressure",
        "XMEAS(08) Reactor Level",
        "XMEAS(09) Reactor Temperature",
        "XMEAS(10) Purge Rate (stream 9)",
        "XMEAS(11) Product Separator Temperature",
        "XMEAS(12) Product Separator Level",
        "XMEAS(13) Prod Separator Pressure",
        "XMEAS(14) Prod Separator Underflow (stream 10)",
        "XMEAS(15) Stripper Level",
        "XMEAS(16) Stripper Pressure",
        "XMEAS(17) Stripper Underflow (stream 11)",
        "XMEAS(18) Stripper Temperature",
        "XMEAS(19) Stripper Steam Flow",
        "XMEAS(20) Compressor Work",
        "XMEAS(21) Reactor CW Outlet Temperature",
        "XMEAS(22) Separator CW Outlet Temperature",
        "XMV(01) D Feed Flow (stream 2)",
        "XMV(02) E Feed Flow (stream 3)",
        "XMV(03) A Feed Flow (stream 1)",
        "XMV(04) A and C Feed Flow (stream 4)",
        "XMV(05) Compressor Recycle Valve",
        "XMV(06) Purge Valve (stream 9)",
        "XMV(07) Separator Pot Liquid Flow (stream 10)",
        "XMV(08) Stripper Liquid Product Flow (stream 11)",
        "XMV(09) Stripper Steam Valve",
        "XMV(10) Reactor CW Flow",
        "XMV(11) Condenser CW Flow",
    ]
    lagged_labels = []
    for i in range(1, lagged_samples + 1):
        for lbl in index_labels:
            # lagged_labels.append(f"{lbl} (t - {i})")
            lagged_labels.append(lbl)
    index_labels += lagged_labels

    for (idv_name, stats_rsfa, conts_rsfa,
         stats_incsfa, conts_incsfa,
         stats_incsfa_svd, conts_incsfa_svd,
         stats_incmssfa, conts_incmssfa) in results:
        """Plot Stats"""

        data_to_plot = [("RSFA", stats_rsfa, conts_rsfa),
                        ("IncSFA", stats_incsfa, conts_incsfa),
                        ("IncSFA SVD", stats_incsfa_svd, conts_incsfa_svd),
                        ("IncMSSFA", stats_incmssfa, conts_incmssfa)]
        for i, (algo, stats, conts) in enumerate(data_to_plot):
            Tc, _, Sc, _ = fd.calculate_crit_values(n, Md[i], Me[i], alpha)

            _f, axs2d = plot_test_stats(algo, idv_name,
                                        stats[0], Tc,
                                        stats[2], Sc)
            axs2d[0].axvline(x=160 - lagged_samples, color='k', linestyle='--')
            axs2d[1].axvline(x=159 - lagged_samples, color='k', linestyle='--')
            plt.savefig(f'plots/CS/{algo}/{algo}_{idv_name}_stats.png',
                        dpi=350)
            plt.close(fig=_f)
            _f = None

            # Get largest contribution for each variable in specified range
            max_T = np.max(conts[0], axis=1)
            max_S = np.max(conts[2], axis=1)

            # Get the order of the contribution values in descending order
            order_T = np.argsort(-1 * max_T)
            order_S = np.argsort(-1 * max_S)
            if n_to_plot < len(order_T):
                order_T = list(order_T)[:n_to_plot]
                order_S = list(order_S)[:n_to_plot]

            # Create plot
            grid_dict = {'width_ratios': [2, 1]}
            _f, ax = plt.subplots(nrows=2, ncols=2, gridspec_kw=grid_dict,
                                  sharex=True)
            _f.set_size_inches(16, 9)

            # Reindex to match up with data sample numbers
            indices_T = [i + lagged_samples + 1 for i in fd_samples]
            indices_T = indices_T[1:]
            T_plot = conts[0][:, 1:]
            indices_S = [i + lagged_samples + 2 for i in fd_samples[:-1]]
            S_plot = conts[2]

            # Plot the contributions
            for index in order_T:
                ax[0, 0].plot(indices_T, T_plot[index, :],
                              **styles[index], label=index_labels[index])
            for index in order_S:
                ax[1, 0].plot(indices_S, S_plot[index, :], **styles[index],
                              label=index_labels[index])

            ax[0, 0].set_title(f"{algo} Top {n_to_plot} $T^2$ "
                               f"Contributors for {idv_name}",
                               fontsize=20)
            ax[0, 0].set_ylabel('$T^2$ Contribution', fontsize=20)
            ax[0, 0].set_xticks(indices_T)
            ax[0, 0].tick_params(axis='both', which='major', labelsize=15)

            ax[1, 0].set_title(f"{algo} Top {n_to_plot} $S^2$ "
                               f"Contributors for {idv_name}",
                               fontsize=20)
            ax[1, 0].set_ylabel('$S^2$ Contribution', fontsize=20)
            ax[1, 0].set_xticks(indices_S)
            ax[1, 0].tick_params(axis='both', which='major', labelsize=15)
            ax[1, 0].set_xlabel('Sample', fontsize=20)

            # Add legend
            lines_labels = [ax.get_legend_handles_labels() for ax in _f.axes]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
            # Find duplicates in labels
            duplicate_labels = []  # List of dupes
            for i in range(len(labels)):
                if i in duplicate_labels:
                    # Don't search of dupes of this
                    # value if it's index is a dupe
                    continue
                for j in range(len(labels)):
                    is_dupe = labels[i] in labels[j]
                    is_same = i == j
                    is_in_dupes = j in duplicate_labels
                    if is_dupe and not is_same and not is_in_dupes:
                        duplicate_labels.append(j)
            # Remove dupes in reverse sort order to keep indices accurate
            duplicate_labels.sort()
            duplicate_labels.reverse()
            for j in duplicate_labels:
                labels.pop(j)
                lines.pop(j)
            # Sort legend alphabetically
            labels, lines = zip(*sorted(zip(labels, lines)))
            labels = list(labels)
            lines = list(lines)
            for i in range(len(lines)):
                lines[i] = Line2D([0], [0], color=lines[i]._color,
                                  markersize=lines[i]._markersize,
                                  marker=lines[i]._marker._marker,
                                  fillstyle=lines[i]._marker._fillstyle,
                                  linestyle='None')
            # Add lines for lagged samples
            for i in range(min(lagged_samples + 1, 3)):
                if i > 0:
                    time = f't - {i}'
                else:
                    time = 't'
                labels.append(f'Time = {time}')
                lines.append(Line2D([0], [0],
                             color='black',
                             linestyle=linestyles[i]))
            gs = ax[0, 1].get_gridspec()
            ax[0, 1].remove()
            ax[1, 1].remove()
            axleg = _f.add_subplot(gs[:, 1])
            axleg.legend(lines, labels, loc='center', frameon=False,
                         handlelength=4, fontsize=15)
            axleg.axis('off')

            _f.set_tight_layout(True)

            plt.savefig(f'plots/CS/{algo}/{algo}_{idv_name}_FD.png', dpi=350)
            plt.close(fig=_f)
