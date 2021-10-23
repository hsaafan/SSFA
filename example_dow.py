import src.sfamanopt.ssfa as ssfa
import src.sfamanopt.fault_diagnosis as fd

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import tepimport
import pypcfd.plotting as fdplt

if __name__ == "__main__":
    alpha = 0.01
    Md = 45
    lagged_samples = 2
    sample = 500  # Contribution plot sample
    n_to_plot = 5  # Contribution plot number of variables to plot
    """Import Data"""
    X = np.loadtxt(open("dow/training.csv", "r"), delimiter=",").T
    T = np.loadtxt(open("dow/testing.csv", "r"), delimiter=",").T
    X = tepimport.add_lagged_samples(X, lagged_samples)
    T = tepimport.add_lagged_samples(T, lagged_samples)

    """Data Preprocessing"""
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

    X_test = ((T - X_mean) / X_std)
    Y = (W.T @ X_test)
    test_stats = fd.calculate_test_stats(Y, Md, Omega_inv)
    contributions = fd.calculate_fault_contributions(X_test, W.T,
                                                     Omega_inv, Md)

    Td, Te, Sd, Se = test_stats
    Td_cont, Te_cont, Sd_cont, Se_cont = contributions
    Tdc, Tec, Sdc, Sec = fd.calculate_crit_values(T.shape[1], Md, Me, alpha)

    index_labels = [
        "x1:Primary Column Reflux Flow",
        "x2:Primary Column Tails Flow",
        "x3:Input to Primary Column Bed 3 Flow",
        "x4:Input to Primary Column Bed 2 Flow",
        "x5:Primary Column Feed Flow from Feed Column",
        "x6:Primary Column Make Flow",
        "x7:Primary Column Base Level",
        "x8:Primary Column Reflux Drum Pressure",
        "x9:Primary Column Condenser Reflux Drum Level",
        "x10:Primary Column Bed1 DP",
        "x11:Primary Column Bed2 DP",
        "x12:Primary Column Bed3 DP",
        "x13:Primary Column Bed4 DP",
        "x14:Primary Column Base Pressure",
        "x15:Primary Column Head Pressure",
        "x16:Primary Column Tails Temperature",
        "x17:Primary Column Tails Temperature 1",
        "x18:Primary Column Bed 4 Temperature",
        "x19:Primary Column Bed 3 Temperature",
        "x20:Primary Column Bed 2 Temperature",
        "x21:Primary Column Bed 1 Temperature",
        "x22: Secondary Column Base Concentration",
        "x23: Flow from Input to Secondary Column",
        "x24: Secondary Column Tails Flow",
        "x25: Secondary Column Tray DP",
        "x26: Secondary Column Head Pressure",
        "x27: Secondary Column Base Pressure",
        "x28: Secondary Column Base Temperature",
        "x29: Secondary Column Tray 3 Temperature",
        "x30: Secondary Column Bed 1 Temperature",
        "x31: Secondary Column Bed 2 Temperature",
        "x32: Secondary Column Tray 2 Temperature",
        "x33: Secondary Column Tray 1 Temperature",
        "x34: Secondary Column Tails Temperature",
        "x35: Secondary Column Tails Concentration",
        "x36: Feed Column Recycle Flow",
        "x37: Feed Column Tails Flow to Primary Column",
        "x38: Feed Column Calculated DP",
        "x39: Feed Column Steam Flow",
        "x40: Feed Column Tails Flow",
        "Avg_Reactor_Outlet_Impurity",
        "Avg_Delta_Composition Primary Column",
        "y:Impurity",
        "Primary Column Reflux/Feed Ratio",
        "Primary Column Make/Reflux Ratio"
        ]
    lagged_labels = []
    for i in range(1, lagged_samples + 1):
        for lbl in index_labels:
            lagged_labels.append(f"{lbl} (t - {i})")
    index_labels += lagged_labels

    fd.plot_test_stats(f"DOW_stats.png",
                       f"DOW ($\\alpha$={alpha}, $M_d$={Md}, $M_e$={Me})",
                       Td, Te, Sd, Se, Tdc, Tec, Sdc, Sec)
    fdplt.plot_contributions(f"DOW_Td_Contributions.png",
                             f"$T_d^2$ Contributions Sample {sample}",
                             Td_cont[:, sample], n_to_plot, index_labels)
    fdplt.plot_contributions(f"DOW_Te_Contributions.png",
                             f"$T_e^2$ Contributions Sample {sample}",
                             Te_cont[:, sample], n_to_plot, index_labels)
    fdplt.plot_contributions(f"DOW_Sd_Contributions.png",
                             f"$S_d^2$ Contributions Sample {sample}",
                             Sd_cont[:, sample], n_to_plot, index_labels)
    fdplt.plot_contributions(f"DOW_Se_Contributions.png",
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
