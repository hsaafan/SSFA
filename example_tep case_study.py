"""
Looks at a sample range, and plots the variable contributions over that range.
Variables are chosen by taking the maximum value of each variable over that
range and then picking the largest n_to_plot variables.
"""
import src.sfamanopt.ssfa as ssfa
import src.sfamanopt.paper_ssfa as oldssfa
import src.sfamanopt.fault_diagnosis as fd

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import tepimport

if __name__ == "__main__":
    alpha = 0.01
    Md = 55
    lagged_samples = 2
    fd_method = 'CDC'
    fd_samples = [i for i in range(157, 163)]  # Contribution plot samples
    n_to_plot = 10  # Contribution plot number of variables to plot
    idv = [4, 11]
    # Linestyles for the n_to_plot variables
    colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
              u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf',
              "black"]
    marker_styles = ['o', 'v', '^', '<', '>', 's', 'D', 'X', 'p', 'P', 'H']
    fill_styles = ['full', 'left', 'top']
    linestyles = ['--', '-.', ':']
    styles = []
    for L in linestyles:
        for f in fill_styles:
            for c, m in zip(colors, marker_styles):
                styles.append(dict(marker=m, linestyle=L,
                                   fillstyle=f, color=c,
                                   alpha=0.5, markersize=15))
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
    W, _, _, _ = ssfa_object.run(X, m)

    """Old SSFA from paper"""
    paper_ssfa_object = oldssfa.PaperSSFA()
    W_old, _, _, _ = paper_ssfa_object.run(X, m, mu=5)

    """Original SFA"""
    U, Lam, UT = np.linalg.svd(np.cov(X))
    Q = U @ np.diag(Lam ** -(1/2))
    Z = Q.T @ X
    Z_dot = Z[:, 1:] - Z[:, :-1]
    P, Omega, PT = np.linalg.svd(np.cov(Z_dot))
    W_orig = Q @ P
    Omega_inv_orig = np.diag(Omega ** -1)

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
    Lambda_inv_old = np.linalg.pinv(W_old.T @ W_old)

    results = []
    for name, test_data in tests:
        X_test = ((test_data - X_mean) / X_std)
        X_cont = X_test[:, fd_samples]

        Y = (W.T @ X_test)
        stats = fd.calculate_test_stats(Y, Md, Omega_inv)
        conts = fd.calculate_fault_contributions(X_cont, W.T, Omega_inv,
                                                 Md, fd_method)

        Y_old = (W_old.T @ X_test)
        stats_old = fd.calculate_test_stats(Y_old, Md, Omega_inv_old)
        conts_old = fd.calculate_fault_contributions(X_cont, W_old.T,
                                                     Omega_inv_old,
                                                     Md, fd_method)

        # Recalculate Td for the code from the paper
        for i in range(n_test):
            stats_old[0][i] = (Y_old[:Md, i].T @ Lambda_inv_old[:Md, :Md]
                               @ Y_old[:Md, i])
            stats_old[1][i] = (Y_old[Md:, i].T @ Lambda_inv_old[Md:, Md:]
                               @ Y_old[Md:, i])

        Y_orig = (W_orig.T @ X_test)
        stats_orig = fd.calculate_test_stats(Y_orig, Md, Omega_inv_orig)
        conts_orig = fd.calculate_fault_contributions(X_cont, W_orig.T,
                                                      Omega_inv_orig,
                                                      Md, fd_method)

        results.append((name,
                        stats, conts,
                        stats_old, conts_old,
                        stats_orig, conts_orig))

    Tdc, Tec, Sdc, Sec = fd.calculate_crit_values(n_test, Md, Me, alpha)

    index_labels = [
        "XMEAS(01) A Feed  (stream 1) kscmh",
        "XMEAS(02) D Feed  (stream 2) kg/hr",
        "XMEAS(03) E Feed  (stream 3) kg/hr",
        "XMEAS(04) A and C Feed  (stream 4) kscmh",
        "XMEAS(05) Recycle Flow  (stream 8) kscmh",
        "XMEAS(06) Reactor Feed Rate  (stream 6) kscmh",
        "XMEAS(07) Reactor Pressure kPa gauge",
        "XMEAS(08) Reactor Level %",
        "XMEAS(09) Reactor Temperature Deg C",
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
        "XMV(01) D Feed Flow (stream 2)",
        "XMV(02) E Feed Flow (stream 3)",
        "XMV(03) A Feed Flow (stream 1)",
        "XMV(04) A and C Feed Flow (stream 4)",
        "XMV(05) Compressor Recycle Valve",
        "XMV(06) Purge Valve (stream 9)",
        "XMV(07) Separator Pot Liquid Flow (stream 10)",
        "XMV(08) Stripper Liquid Product Flow (stream 11)",
        "XMV(09) Stripper Steam Valve",
        "XMV(10) Reactor Cooling Water Flow",
        "XMV(11) Condenser Cooling Water Flow",
    ]
    lagged_labels = []
    for i in range(1, lagged_samples + 1):
        for lbl in index_labels:
            # lagged_labels.append(f"{lbl} (t - {i})")
            lagged_labels.append(lbl)
    index_labels += lagged_labels

    for (name, stats, conts, stats_old,
         conts_old, stats_orig, conts_orig) in results:
        """Plot Stats"""
        _f, axs2d = plt.subplots(nrows=4, ncols=1, sharex=True)
        _f.set_size_inches(10.5, 9)

        Td_plot = axs2d[0]
        Td_plot.set_title(f"{name} Test Statistics")
        Td_plot.set_ylabel("$T^2_d$")
        Td_plot.plot(stats_orig[0], label='SFA')
        Td_plot.plot(stats_old[0], label='Sparse SFA')
        Td_plot.plot(stats[0], label='Manifold Sparse SFA')
        Td_plot.plot([Tdc] * len(stats[0]))
        Td_plot.legend(loc='upper left')

        Te_plot = axs2d[1]
        Te_plot.set_ylabel("$T^2_e$")
        Te_plot.plot(stats_orig[1], label='SFA')
        Te_plot.plot(stats_old[1], label='Sparse SFA')
        Te_plot.plot(stats[1], label='Manifold Sparse SFA')
        Te_plot.plot([Tec] * len(stats[1]))
        Te_plot.legend(loc='upper left')

        Sd_plot = axs2d[2]
        Sd_plot.set_ylabel("$S^2_d$")
        Sd_plot.plot(stats_orig[2], label='SFA')
        Sd_plot.plot(stats_old[2], label='Sparse SFA')
        Sd_plot.plot(stats[2], label='Manifold Sparse SFA')
        Sd_plot.plot([Sdc] * len(stats[2]))
        Sd_plot.legend(loc='upper left')

        Se_plot = axs2d[3]
        Se_plot.set_ylabel("$S^2_e$")
        Se_plot.plot(stats_orig[3], label='SFA')
        Se_plot.plot(stats_old[3], label='Sparse SFA')
        Se_plot.plot(stats[3], label='Manifold Sparse SFA')
        Se_plot.plot([Sec] * len(stats[3]))
        Se_plot.legend(loc='upper left')
        Se_plot.set_xlabel("Sample")

        _f.set_tight_layout(True)
        plt.savefig(f'plots/CS/{name}_stats.png', dpi=350)
        plt.close(fig=_f)
        _f = None

        Td_cont, Te_cont, Sd_cont, Se_cont = conts
        Td_cont_old, Te_cont_old, Sd_cont_old, Se_cont_old = conts_old
        Td_cont_orig, Te_cont_orig, Sd_cont_orig, Se_cont_orig = conts_orig
        for s_name, c, cd, cg in [('Td', Td_cont, Td_cont_old, Td_cont_orig),
                                  ('Te', Te_cont, Te_cont_old, Te_cont_orig),
                                  ('Sd', Sd_cont, Sd_cont_old, Sd_cont_orig),
                                  ('Se', Se_cont, Se_cont_old, Se_cont_orig)]:
            # Get largest contribution for each variable in specified range
            max_c = np.max(c, axis=1)
            max_cd = np.max(cd, axis=1)
            max_cg = np.max(cg, axis=1)

            # Get the order of the contribution values in descending order
            c_order = np.argsort(-1 * max_c)
            cd_order = np.argsort(-1 * max_cd)
            cg_order = np.argsort(-1 * max_cg)

            # Create plot
            _f, ax = plt.subplots(nrows=2, ncols=2)
            _f.set_size_inches(21, 14)
            # Reindex to match up with data sample numbers
            if s_name[0] == "S":
                indices = [i + lagged_samples + 2 for i in fd_samples[:-1]]
            else:
                indices = [i + lagged_samples + 1 for i in fd_samples]

            # Plot the contributions
            for i in range(min(n_to_plot, len(cg_order))):
                index = cg_order[i]
                ax[0, 0].plot(indices,
                              cg[index, :],
                              **styles[index],
                              label=index_labels[index])
            for i in range(min(n_to_plot, len(cd_order))):
                index = cd_order[i]
                ax[0, 1].plot(indices,
                              cd[index, :],
                              **styles[index],
                              label=index_labels[index])
            for i in range(min(n_to_plot, len(c_order))):
                index = c_order[i]
                ax[1, 0].plot(indices,
                              c[index, :],
                              **styles[index],
                              label=index_labels[index])

            for sbplot in [ax[0, 0], ax[0, 1], ax[1, 0]]:
                sbplot.set_ylabel('Contribution', fontsize=20)
                sbplot.set_xlabel('Sample', fontsize=20)
                sbplot.set_xticks(indices)
                sbplot.tick_params(axis='both', which='major', labelsize=15)
            # ax[0, 0].set_ylabel('Contribution')
            # ax[1, 0].set_ylabel('Contribution')
            # for i in range(3):
            #     ax[i].set_xlabel('Sample')
            #     ax[i].set_xticks(indices)
            ax[0, 0].set_title("SFA", x=0.01, y=0.9,
                               loc='left', fontsize=20)
            ax[0, 1].set_title("Sparse SFA", x=0.01, y=0.9,
                               loc='left', fontsize=20)
            ax[1, 0].set_title("Manifold Sparse SFA", x=0.01, y=0.9,
                               loc='left', fontsize=20)
            title = (f'{name}: Top {n_to_plot} Contributing Variables to '
                     f'${s_name[0]}_{s_name[1]}^2$ For Samples '
                     f'{indices[0]} - {indices[-1]}')
            _f.suptitle(title, fontsize=20)
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
            ax[1, 1].legend(lines, labels,
                            loc='center',
                            frameon=False,
                            handlelength=4,
                            fontsize=15)
            ax[1, 1].axis('off')
            _f.subplots_adjust(hspace=0.2)

            plt.savefig(f'plots/CS/{name}_{s_name}.png', dpi=350)
            plt.close(fig=_f)
