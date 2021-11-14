import src.sfamanopt.ssfa as ssfa
import src.sfamanopt.fault_diagnosis as fd

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

import tepimport
import pypcfd.plotting as fdplt


def diagnosis_to_csv_line(idv: str, sample: int, stat: str, n: int,
                          contribs: list, labels: list) -> None:
    order = np.argsort(-1 * contribs[:, sample])
    ordered_contirbs = contribs[:, sample][order]
    pcnt_contribs = ordered_contirbs / np.sum(ordered_contirbs)
    txt_output = f"{idv},{stat},{sample},"
    for i in range(n):
        txt_output += f"{labels[order[i]]},{pcnt_contribs[i]*100:.2f},"
    with open('plots/contribs.csv', 'a') as f:
        f.write(f"{txt_output[:-1]}\n")


if __name__ == "__main__":
    alpha = 0.01
    Md = 20
    lagged_samples = 0
    fd_method = 'CDC'
    fd_samples = []  # Contribution plot samples
    n_to_plot = 5  # Contribution plot number of variables to plot
    use_original_sfa = True
    # fd_samples = [(159, 159, 157, 157), (159, 159, 157, 157)]
    fd_samples = [(161, 161, 160, 160), (161, 161, 160, 160)]
    """Import Data"""
    X, _, _, _, _ = tepimport.import_tep_sets(lagged_samples)
    test_sets = tepimport.import_sets([4, 5], skip_training=True)
    m = X.shape[0]
    n = X.shape[1]
    X_mean = np.mean(X, axis=1).reshape((-1, 1))
    X = X - X_mean
    X_std = np.std(X, axis=1).reshape((-1, 1))
    X_std = np.ones_like(X_mean)
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

    if use_original_sfa:
        U, Lam, UT = np.linalg.svd(np.cov(X))
        Q = U @ np.diag(Lam ** -(1/2))
        Z = Q.T @ X
        Z_dot = Z[:, 1:] - Z[:, :-1]
        P, Omega, PT = np.linalg.svd(np.cov(Z_dot))
        W = Q @ P
        Omega_inv = np.diag(Omega ** -1)

    ignored_var = list(range(22, 41))
    tests = []
    for name, data in test_sets:
        data = np.delete(data, ignored_var, axis=0)
        data = tepimport.add_lagged_samples(data, lagged_samples)
        tests.append((name, data))
    n_test = tests[0][1].shape[1]

    results = []
    for name, test_data in tests:
        X_test = ((test_data - X_mean) / X_std)
        Y = (W.T @ X_test)
        test_stats = fd.calculate_test_stats(Y, Md, Omega_inv)
        contributions = fd.calculate_fault_contributions(X_test, W.T,
                                                         Omega_inv, Md,
                                                         fd_method)
        results.append((name, *test_stats, *contributions))
        # fd_samples.append((int(np.argmax(test_stats[0])),
        #                    int(np.argmax(test_stats[1])),
        #                    int(np.argmax(test_stats[2])),
        #                    int(np.argmax(test_stats[3]))))

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

    open('plots/contribs.csv', 'w').close()
    for name, Td, Te, Sd, Se, Td_cont, Te_cont, Sd_cont, Se_cont in results:
        Td_sample, Te_sample, Sd_sample, Se_sample = fd_samples.pop(0)
        fd.plot_test_stats(f"plots/{name}_stats.png",
                           f"{name} ($\\alpha$={alpha}, "
                           f"$M_d$={Md}, $M_e$={Me})",
                           Td, Te, Sd, Se, Tdc, Tec, Sdc, Sec)
        fdplt.plot_contributions(f"plots/FD/{name}_Td_Contributions.png",
                                 f"$T_d^2$ Contributions Sample {Td_sample}",
                                 Td_cont[:, Td_sample],
                                 n_to_plot,
                                 index_labels)
        fdplt.plot_contributions(f"plots/FD/{name}_Te_Contributions.png",
                                 f"$T_e^2$ Contributions Sample {Te_sample}",
                                 Te_cont[:, Te_sample],
                                 n_to_plot,
                                 index_labels)
        fdplt.plot_contributions(f"plots/FD/{name}_Sd_Contributions.png",
                                 f"$S_d^2$ Contributions Sample {Sd_sample}",
                                 Sd_cont[:, Sd_sample],
                                 n_to_plot,
                                 index_labels)
        fdplt.plot_contributions(f"plots/FD/{name}_Se_Contributions.png",
                                 f"$S_e^2$ Contributions Sample {Se_sample}",
                                 Se_cont[:, Se_sample],
                                 n_to_plot,
                                 index_labels)
        diagnosis_to_csv_line(name, Td_sample, 'Td', 5, Td_cont, index_labels)
        diagnosis_to_csv_line(name, Te_sample, 'Te', 5, Te_cont, index_labels)
        diagnosis_to_csv_line(name, Sd_sample, 'Sd', 5, Sd_cont, index_labels)
        diagnosis_to_csv_line(name, Se_sample, 'Se', 5, Se_cont, index_labels)

    plt.figure()
    # plt.subplot(2, 1, 1)
    plt.plot(speeds[order])
    plt.ylabel("Feature Speed")
    plt.xlabel("Slow Features")
    # plt.subplot(2, 1, 2)
    # plt.plot([np.count_nonzero(W[:, i] == 0) for i in range(m)])
    # plt.ylabel("Number of zero values")

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.title("Sparse Values")
    plt.plot(sparsity_values)
    plt.subplot(3, 1, 2)
    plt.title("Relative Error")
    plt.plot(relative_errors[1:])
    plt.yscale('log')
    plt.subplot(3, 1, 3)
    plt.title("Cost Function")
    plt.plot(cost_values)
    plt.yscale('log')

    plt.figure()
    plt.imshow(np.abs(W.T))
    plt.colorbar()

    sparse_W_vals = int(sparsity_values[-1] * np.size(W))
    print(f'0 values of W: {sparse_W_vals} / {np.size(W)}')
    plt.show()
