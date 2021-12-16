"""
Looks at a sample range, and plots the variable contributions over that range.
Variables are chosen by taking the maximum value of each variable over that
range and then picking the largest n_to_plot variables.
"""
import src.sfamanopt.mssfa as mssfa
import src.sfamanopt.ssfa as ssfa
import src.sfamanopt.sfa as sfa
import src.sfamanopt.fault_diagnosis as fd

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import SparsePCA

import tepimport

if __name__ == "__main__":
    load_ssfa = True
    alpha = 0.01
    Md = 55
    lagged_samples = 2
    fd_method = 'CDC'
    fd_samples = [i for i in range(202, 213)]  # Contribution plot samples
    n_to_plot = 5  # Contribution plot number of variables to plot
    idv = [11]
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
    Me = m - Md

    """Train Model"""
    sfa_object = sfa.SFA()
    W_sfa, Omega_inv_sfa = sfa_object.run(X, Md)

    ssfa_object = ssfa.SSFA()
    if load_ssfa:
        with open('ssfa_matrix.npy', 'rb') as f:
            W_ssfa = np.load(f)
            Omega_inv_ssfa = np.load(f)
    else:
        W_ssfa, Omega_inv_ssfa, _, _, _ = ssfa_object.run(X, Md)
    Lambda_inv_ssfa = np.linalg.pinv(W_ssfa.T @ W_ssfa)

    spca = SparsePCA(n_components=Md, max_iter=500, tol=1e-6)
    spca.fit(X.T)
    print(f"SPCA converged in {spca.n_iter_} iterations")
    P = spca.components_.T
    P_d = P[:, :Md]
    P_e = P[:, Md:]
    scores_d = X.T @ P_d
    scores_e = X.T @ P_e
    gamma_inv_d = np.linalg.inv(np.cov(scores_d.T))
    gamma_inv_e = np.linalg.inv(np.cov(scores_e.T))

    mssfa_object = mssfa.MSSFA("chol", "l1")
    W_mssfa, Omega_inv_mssfa, _, _, _ = mssfa_object.run(X, Md)

    results = []
    for name, test_data in tests:
        X_test = ((test_data - X_mean) / X_std)
        X_cont = X_test[:, fd_samples]

        Y_sfa = (W_sfa.T @ X_test)
        stats_sfa = fd.calculate_test_stats(Y_sfa, Md, Omega_inv_sfa)
        conts_sfa = fd.calculate_fault_contributions(X_cont, W_sfa.T,
                                                     Omega_inv_sfa,
                                                     Md, fd_method)

        Y_ssfa = (W_ssfa.T @ X_test)
        stats_ssfa = fd.calculate_test_stats(Y_ssfa, Md, Omega_inv_ssfa)
        conts_ssfa = fd.calculate_fault_contributions(X_cont, W_ssfa.T,
                                                      Omega_inv_ssfa,
                                                      Md, fd_method)

        stats_spca = fd.calculate_test_stats_pca(X_test.T, P, gamma_inv_d,
                                                 gamma_inv_e, Md)
        conts_spca = fd.calculate_fault_contributions_pca(X_cont.T, P,
                                                          gamma_inv_d,
                                                          gamma_inv_e, Md)

        Y_mssfa = (W_mssfa.T @ X_test)
        stats_mssfa = fd.calculate_test_stats(Y_mssfa, Md, Omega_inv_mssfa)
        conts_mssfa = fd.calculate_fault_contributions(X_cont, W_mssfa.T,
                                                       Omega_inv_mssfa,
                                                       Md, fd_method)
        for i in range(n_test):
            stats_ssfa[0][i] = (Y_ssfa[:Md, i].T @ Lambda_inv_ssfa[:Md, :Md]
                                @ Y_ssfa[:Md, i])
            stats_ssfa[1][i] = (Y_ssfa[Md:, i].T @ Lambda_inv_ssfa[Md:, Md:]
                                @ Y_ssfa[Md:, i])

        results.append((name,
                        stats_sfa, conts_sfa,
                        stats_ssfa, conts_ssfa,
                        stats_spca, conts_spca,
                        stats_mssfa, conts_mssfa))

    Tdc, Tec, Sdc, Sec = fd.calculate_crit_values(n_test, Md, Me, alpha)
    Tdc_pca, Tec_pca, SPEd, SPEe = fd.calculate_crit_values_pca(X.T, P, n, Md,
                                                                Me, alpha)

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

    for (idv_name, stats_sfa, conts_sfa, stats_ssfa, conts_ssfa,
         stats_spca, conts_spca, stats_mssfa, conts_mssfa) in results:
        """Plot Stats"""
        _f, axs2d = plt.subplots(nrows=2, ncols=1, sharex=True)
        _f.set_size_inches(16, 9)

        Td_plot = axs2d[0]
        Td_plot.set_title(f"{idv_name} Test Statistics", fontsize=20)
        Td_plot.set_ylabel("$T^2$", fontsize=20)
        Td_plot.plot(stats_sfa[0], label='SFA')
        Td_plot.plot(stats_ssfa[0], label='Sparse SFA')
        Td_plot.plot(stats_mssfa[0], label='Manifold Sparse SFA')
        Td_plot.plot([Tdc] * len(stats_sfa[0]), 'k')
        Td_plot.legend(loc='upper left', handlelength=4, fontsize=15)
        Td_plot.tick_params(axis='both', which='major', labelsize=15)

        Sd_plot = axs2d[1]
        Sd_plot.set_ylabel("$S^2$", fontsize=20)
        Sd_plot.plot(stats_sfa[2], label='SFA')
        Sd_plot.plot(stats_ssfa[2], label='Sparse SFA')
        Sd_plot.plot(stats_mssfa[2], label='Manifold Sparse SFA')
        Sd_plot.plot([Sdc] * len(stats_sfa[2]), 'k')
        Sd_plot.legend(loc='upper left', handlelength=4, fontsize=15)
        Sd_plot.set_xlabel("Sample", fontsize=20)
        Sd_plot.tick_params(axis='both', which='major', labelsize=15)

        _f.set_tight_layout(True)
        plt.savefig(f'plots/CS/{idv_name}_stats.png', dpi=350)
        plt.close(fig=_f)
        _f = None

        _f_spca, axs2d_spca = plt.subplots(nrows=2, ncols=1, sharex=True)
        _f_spca.set_size_inches(16, 9)

        Td_spca_plot = axs2d_spca[0]
        Td_spca_plot.set_title(f"{idv_name} Test Statistics", fontsize=20)
        Td_spca_plot.set_ylabel("$T^2$", fontsize=20)
        Td_spca_plot.plot(stats_spca[0], label='Sparse PCA')
        Td_spca_plot.plot([Tdc_pca] * len(stats_spca[0]), 'k')
        Td_spca_plot.legend(loc='upper left', handlelength=4, fontsize=15)
        Td_spca_plot.tick_params(axis='both', which='major', labelsize=15)

        Sd_spca_plot = axs2d_spca[1]
        Sd_spca_plot.set_ylabel("SPE", fontsize=20)
        Sd_spca_plot.plot(stats_spca[2], label='Sparse PCA')
        Sd_spca_plot.plot([SPEd] * len(stats_sfa[2]), 'k')
        Sd_spca_plot.legend(loc='upper left', handlelength=4, fontsize=15)
        Sd_spca_plot.set_xlabel("Sample", fontsize=20)
        Sd_spca_plot.tick_params(axis='both', which='major', labelsize=15)

        _f_spca.set_tight_layout(True)
        plt.savefig(f'plots/CS/{idv_name}_stats_spca.png', dpi=350)
        plt.close(fig=_f_spca)
        _f_spca = None

        for name, conts in zip(["SFA", "SSFA", "SPCA", "MSSFA"],
                               [conts_sfa, conts_ssfa,
                                conts_spca, conts_mssfa]):
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
            if name == "SPCA":
                indices_S = indices_T.copy()
                S_plot = conts[2][:, 1:]
            else:
                indices_S = [i + lagged_samples + 2 for i in fd_samples[:-1]]
                S_plot = conts[2]

            # Plot the contributions
            for index in order_T:
                ax[0, 0].plot(indices_T, T_plot[index, :],
                              **styles[index], label=index_labels[index])
            for index in order_S:
                ax[1, 0].plot(indices_S, S_plot[index, :], **styles[index],
                              label=index_labels[index])

            ax[0, 0].set_title(f"{name} Top {n_to_plot} $T^2$ "
                               f"Contributors for {idv_name}",
                               fontsize=20)
            ax[0, 0].set_ylabel('$T^2$ Contribution', fontsize=20)
            ax[0, 0].set_xticks(indices_T)
            ax[0, 0].tick_params(axis='both', which='major', labelsize=15)

            if name == "SPCA":
                ax[1, 0].set_title(f"{name} Top {n_to_plot} SPE "
                                   f"Contributors for {idv_name}",
                                   fontsize=20)
                ax[1, 0].set_ylabel('SPE Contribution', fontsize=20)
            else:
                ax[1, 0].set_title(f"{name} Top {n_to_plot} $S^2$ "
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

            plt.savefig(f'plots/CS/{idv_name}_{name}_FD.png', dpi=350)
            plt.close(fig=_f)
