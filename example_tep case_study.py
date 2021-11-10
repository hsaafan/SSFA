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
    Md = 91
    lagged_samples = 2
    fd_method = 'CDC'
    fd_samples = [i for i in range(150, 170)]  # Contribution plot samples
    n_to_plot = 10  # Contribution plot number of variables to plot
    # Linestyles for the n_to_plot variables
    linestyles = ['--s', '--o', '--<', '-->', '--H',
                  ':s', ':o', ':<', ':>', ':H']
    """Import Data"""
    ignored_var = list(range(22, 41))
    X = tepimport.import_sets([0], skip_test=True)[0][1]
    X = tepimport.add_lagged_samples(np.delete(X, ignored_var, axis=0),
                                     lagged_samples)

    test_sets = tepimport.import_sets([4, 5], skip_training=True)
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

    results = []
    for name, test_data in tests:
        X_test = ((test_data - X_mean) / X_std)
        Y = (W.T @ X_test)
        test_stats = fd.calculate_test_stats(Y, Md, Omega_inv)
        contributions = fd.calculate_fault_contributions(X_test, W.T,
                                                         Omega_inv, Md,
                                                         fd_method)
        results.append((name, *test_stats, *contributions))

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

    for name, Td, Te, Sd, Se, Td_cont, Te_cont, Sd_cont, Se_cont in results:
        for stat_name, cont in [('Td', Td_cont), ('Te', Te_cont),
                                ('Sd', Sd_cont), ('Se', Se_cont)]:
            cont_range = cont[:, fd_samples]
            largest_cont_over_range = np.max(cont_range, axis=1)
            order = np.argsort(-1 * largest_cont_over_range)
            _f, ax = plt.subplots()
            _f.set_size_inches(16, 9)
            for i in range(min(n_to_plot, len(order))):
                index = order[i]
                ax.plot(fd_samples,
                        cont_range[index, :],
                        linestyles[i],
                        label=index_labels[index])
            ax.legend(loc='upper left')
            ax.set_xlabel('Sample')
            ax.set_xticks(fd_samples)
            ax.set_ylabel('Contribution')
            title = (f'{name}: Top {n_to_plot} Contributing Variables to '
                     f'${stat_name[0]}_{stat_name[1]}^2$ For Samples '
                     f'{fd_samples[0]} - {fd_samples[-1]}')
            ax.set_title(title)
            plt.savefig(f'plots/CS/{name}_{stat_name}.png', dpi=350)
            plt.close(fig=_f)
