"""
Looks at a sample range, and plots the variable contributions over that range.
Variables are chosen by taking the maximum value of each variable over that
range and then picking the largest n_to_plot variables.
"""
import src.sfamanopt.mssfa as mssfa
import src.sfamanopt.ssfa as ssfa
import src.sfamanopt.sfa as sfa
import src.sfamanopt.fault_diagnosis as fd
import src.sfamanopt.load_cva as cva

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import SparsePCA


if __name__ == "__main__":
    load_models = False
    alpha = 0.01
    Md = 12
    fd_method = 'CDC'
    fd_samples = [i for i in range(649, 660)]  # Contribution plot samples
    n_to_plot = 5  # Contribution plot number of variables to plot
    fault_index = 0
    subfault_index = 1
    # Linestyles for the n_to_plot variables
    markers = [(u'#1f77b4', 'o', 'left'),       # Var 01
               (u'#ff7f0e', 'v', 'full'),       # Var 02
               (u'#2ca02c', '^', 'full'),       # Var 03
               (u'#7f7f7f', 'X', 'top'),        # Var 04
               (u'#8c564b', 's', 'left'),       # Var 05
               (u'#e377c2', 'D', 'full'),       # Var 06
               (u'#7f7f7f', 'X', 'left'),       # Var 07
               (u'#bcbd22', 'p', 'full'),       # Var 08
               (u'#d62728', '<', 'left'),       # Var 09
               ("black", 'H', 'top'),           # Var 10
               (u'#ff7f0e', 'v', 'top'),        # Var 11
               (u'#ff7f0e', 'v', 'left'),       # Var 12
               (u'#1f77b4', 'o', 'full'),       # Var 13
               (u'#2ca02c', '^', 'left'),       # Var 14
               (u'#17becf', 'P', 'full'),       # Var 15
               (u'#d62728', '<', 'full'),       # Var 16
               (u'#9467bd', '>', 'full'),       # Var 17
               (u'#8c564b', 's', 'full'),       # Var 18
               (u'#e377c2', 'D', 'left'),       # Var 19
               (u'#7f7f7f', 'X', 'full'),       # Var 20
               (u'#bcbd22', 'p', 'left'),       # Var 21
               (u'#17becf', 'P', 'full'),       # Var 22
               ("black", 'H', 'full'),          # Var 23
               (u'#1f77b4', 'o', 'top'),        # Var 24
               ]
    linestyles = ['--', '-.', ':']
    styles = []
    for L in linestyles:
        for c, m, f in markers:
            styles.append(dict(marker=m, linestyle=L,
                               fillstyle=f, color=c,
                               alpha=1, markersize=15))
    """Import Data"""
    """Import Data"""
    data_sets = cva.import_sets()
    X = np.hstack((data_sets[0][1][0][0],
                   data_sets[0][1][1][0],
                   data_sets[0][1][2][0]))

    name, data, f_range = data_sets[0]
    tests = [(f"{name}.{subfault_index+1}", data[0], f_range[subfault_index])]

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
    ssfa_object = ssfa.SSFA()
    mssfa_object = mssfa.MSSFA("chol", "l1")

    if load_models:
        with open('cva_models_matrix.npy', 'rb') as f:
            W_sfa = np.load(f)
            Omega_inv_sfa = np.load(f)
            W_ssfa = np.load(f)
            Omega_inv_ssfa = np.load(f)
            W_mssfa = np.load(f)
            Omega_inv_mssfa = np.load(f)
    else:
        W_sfa, Omega_inv_sfa = sfa_object.run(X, Md)
        W_ssfa, Omega_inv_ssfa, _, _, _ = ssfa_object.run(X, Md)
        W_mssfa, Omega_inv_mssfa, _, _, _ = mssfa_object.run(X, Md)
        with open('cva_models_matrix.npy', 'wb') as f:
            np.save(f, W_sfa)
            np.save(f, Omega_inv_sfa)
            np.save(f, W_ssfa)
            np.save(f, Omega_inv_ssfa)
            np.save(f, W_mssfa)
            np.save(f, Omega_inv_mssfa)
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

    results = []
    for name, test_data, fault_samples in tests:
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
        for i in range(X_test.shape[1]):
            stats_ssfa[0][i] = (Y_ssfa[:Md, i].T @ Lambda_inv_ssfa[:Md, :Md]
                                @ Y_ssfa[:Md, i])
            stats_ssfa[1][i] = (Y_ssfa[Md:, i].T @ Lambda_inv_ssfa[Md:, Md:]
                                @ Y_ssfa[Md:, i])

        results.append((name, fault_samples,
                        stats_sfa, conts_sfa,
                        stats_ssfa, conts_ssfa,
                        stats_spca, conts_spca,
                        stats_mssfa, conts_mssfa))

    Tdc, Tec, Sdc, Sec = fd.calculate_crit_values(n, Md, Me, alpha)
    Tdc_pca, Tec_pca, SPEd, SPEe = fd.calculate_crit_values_pca(X.T, P, n, Md,
                                                                Me, alpha)

    index_labels = [
        'PT312 Air delivery pressure',
        'PT401 Pressure in the bottom of the riser',
        'PT408 Pressure in top of the riser',
        'PT403 Pressure in top separator',
        'PT501 Pressure in 3 phase separator',
        'PT408 Diff. pressure (PT401-PT408)',
        'PT403 Differential pressure over VC404',
        'FT305 Flow rate input air',
        'FT104 Flow rate input water',
        'FT407 Flow rate top riser',
        'LI405 Level top separator',
        'FT406 Flow rate top separator output',
        'FT407 Density top riser',
        'FT406 Density top separator output',
        'FT104 Density water input',
        'FT407 Temperature top riser',
        'FT406 Temperature top separator output',
        'FT104 Temperature water input',
        'LI504 Level gas-liquid 3 phase separator',
        'VC501 Position of valve VC501',
        'VC302 Position of valve VC302',
        'VC101 Position of valve VC101',
        'PO1 Water pump current',
        'PT417 Pressure in mixture zone 2” line'
    ]
    # lagged_labels = []
    # for i in range(1, lagged_samples + 1):
    #     for lbl in index_labels:
    #         # lagged_labels.append(f"{lbl} (t - {i})")
    #         lagged_labels.append(lbl)
    # index_labels += lagged_labels

    for (name, f_samples, stats_sfa, conts_sfa, stats_ssfa, conts_ssfa,
         stats_spca, conts_spca, stats_mssfa, conts_mssfa) in results:
        """Plot Stats"""
        _f, axs2d = plt.subplots(nrows=2, ncols=1, sharex=True)
        _f.set_size_inches(16, 9)

        Td_plot = axs2d[0]
        Td_plot.set_title(f"{name} Test Statistics", fontsize=20)
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
        plt.savefig(f'plots/CVA/{name}_stats.png', dpi=350)
        plt.close(fig=_f)
        _f = None

        _f_spca, axs2d_spca = plt.subplots(nrows=2, ncols=1, sharex=True)
        _f_spca.set_size_inches(16, 9)

        Td_spca_plot = axs2d_spca[0]
        Td_spca_plot.set_title(f"{name} Test Statistics", fontsize=20)
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
        plt.savefig(f'plots/CVA/{name}_stats_spca.png', dpi=350)
        plt.close(fig=_f_spca)
        _f_spca = None

        for model_name, conts in zip(["SFA", "SSFA", "SPCA", "MSSFA"],
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
            # indices_T = [i + lagged_samples + 1 for i in fd_samples]
            indices_T = [i + 2 for i in fd_samples[1:]]
            T_plot = conts[0][:, 1:]
            if model_name == "SPCA":
                indices_S = indices_T.copy()
                S_plot = conts[2][:, 1:]
            else:
                # indices_S = [i + lagged_samples + 2 for i in fd_samples[:-1]]
                indices_S = [i + 2 for i in fd_samples[:-1]]
                S_plot = conts[2]

            # Plot the contributions
            for index in order_T:
                ax[0, 0].plot(indices_T, T_plot[index, :],
                              **styles[index], label=index_labels[index])
            for index in order_S:
                ax[1, 0].plot(indices_S, S_plot[index, :], **styles[index],
                              label=index_labels[index])

            ax[0, 0].set_title(f"{model_name} Top {n_to_plot} $T^2$ "
                               f"Contributors for {name}",
                               fontsize=20)
            ax[0, 0].set_ylabel('$T^2$ Contribution', fontsize=20)
            ax[0, 0].set_xticks(indices_T)
            ax[0, 0].tick_params(axis='both', which='major', labelsize=15)

            if name == "SPCA":
                ax[1, 0].set_title(f"{model_name} Top {n_to_plot} SPE "
                                   f"Contributors for {name}",
                                   fontsize=20)
                ax[1, 0].set_ylabel('SPE Contribution', fontsize=20)
            else:
                ax[1, 0].set_title(f"{model_name} Top {n_to_plot} $S^2$ "
                                   f"Contributors for {name}",
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
            # Add lines for lagged samples
            gs = ax[0, 1].get_gridspec()
            ax[0, 1].remove()
            ax[1, 1].remove()
            axleg = _f.add_subplot(gs[:, 1])
            axleg.legend(lines, labels, loc='center', frameon=False,
                         handlelength=4, fontsize=15)
            axleg.axis('off')

            _f.set_tight_layout(True)

            plt.savefig(f'plots/CVA/{name}_{model_name}_FD.png', dpi=350)
            plt.close(fig=_f)
