import src.sfamanopt.ssfa as ssfa
import src.sfamanopt.fault_diagnosis as fd

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

import tepimport
import pypcfd.plotting as fdplt

if __name__ == "__main__":
    alpha = 0.01
    Md = 95
    lagged_samples = 2
    sample = 500  # Contribution plot sample
    n_to_plot = 5  # Contribution plot number of variables to plot
    """Import Data"""
    X, T0, T4, T5, T10 = tepimport.import_tep_sets(lagged_samples)
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

    n_test = T0.shape[1]
    tests = [("IDV(0)", T0), ("IDV(4)", T4), ("IDV(5)", T5), ("IDV(10)", T10)]

    results = []
    for name, test_data in tests:
        X_test = ((test_data - X_mean) / X_std)
        Y = (W.T @ X_test)
        test_stats = fd.calculate_test_stats(Y, Md, Omega_inv)
        contributions = fd.calculate_fault_contributions(X_test, W.T,
                                                         Omega_inv, Md)
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
        fd.plot_test_stats(f"{name}_stats.png",
                           f"{name} ($\\alpha$={alpha}, "
                           f"$M_d$={Md}, $M_e$={Me})",
                           Td, Te, Sd, Se, Tdc, Tec, Sdc, Sec)
        fdplt.plot_contributions(f"{name}_Td_Contributions.png",
                                 f"$T_d^2$ Contributions Sample {sample}",
                                 Td_cont[:, sample], n_to_plot, index_labels)
        fdplt.plot_contributions(f"{name}_Te_Contributions.png",
                                 f"$T_e^2$ Contributions Sample {sample}",
                                 Te_cont[:, sample], n_to_plot, index_labels)
        fdplt.plot_contributions(f"{name}_Sd_Contributions.png",
                                 f"$S_d^2$ Contributions Sample {sample}",
                                 Sd_cont[:, sample], n_to_plot, index_labels)
        fdplt.plot_contributions(f"{name}_Se_Contributions.png",
                                 f"$S_e^2$ Contributions Sample {sample}",
                                 Se_cont[:, sample], n_to_plot, index_labels)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(speeds[order])
    plt.ylabel("Feature Speed")
    plt.subplot(2, 1, 2)
    plt.plot([np.count_nonzero(W[:, i] == 0) for i in range(m)])
    plt.xlabel("Slow Features")
    plt.ylabel("Number of zero values")

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

    print(W)
    print(f'0 values of W: {np.count_nonzero(W==0)} / {np.size(W)}')
    plt.show()
