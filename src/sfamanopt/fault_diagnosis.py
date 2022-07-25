from typing import Tuple

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pypcfd.fault_diagnosis as fd


def calculate_crit_values(n: int,
                          Md: int,
                          Me: int,
                          alpha: float = 0.01) -> Tuple[float, float,
                                                        float, float]:
    """ Calculate critical values for monitoring

    Parameters
    ----------
    n: int
        Number of training samples
    Md: int
        Number of slow features
    Me: int
        Number of fast features
    alpha: float
        The confidence level to use for the critical values

    Returns
    -------
    T_d_crit: float
        The critical T^2 value for the slowest features
    T_e_crit: float
        The critical T^2 value for the fastest features
    S_d_crit: float
        The critical S^2 value for the slowest features
    S_e_crit: float
        The crtiical S^2 value for the fastest features
    """
    if alpha > 1 or alpha < 0:
        raise ValueError("Confidence level should be between 0 and 1")
    p = 1 - alpha
    hd = (Md * (n - 1) * (n + 1)) / (n * (n - Md))
    he = (Me * (n - 1) * (n + 1)) / (n * (n - Me))
    gd = (Md*(n**2-2*n))/((n-1)*(n-Md-1))
    ge = (Me*(n**2-2*n))/((n-1)*(n-Me-1))

    T_d_crit = hd * stats.f.ppf(p, Md, n-Md)
    T_e_crit = he * stats.f.ppf(p, Me, n-Me)
    S_d_crit = gd * stats.f.ppf(p, Md, n-Md-1)
    S_e_crit = ge * stats.f.ppf(p, Me, n-Me-1)

    return(T_d_crit, T_e_crit, S_d_crit, S_e_crit)


def calculate_crit_values_pca(X: np.ndarray, P: np.ndarray,
                              n: int, Md: int, Me: int,
                              alpha: float = 0.01) -> Tuple[float, float,
                                                            float, float]:
    """ Calculate critical values for monitoring

    Parameters
    ----------
    X: np.ndarray
        An [n X m] data matrix to calculate the statistics for
    P: np.ndarray
        An [m X m] matrix of components
    Md: int
        The number of components to keep, 1 <= Md <= m
    Me: int
        The number of components removed, 1 <= Me <= m
    alpha: float
        The confidence level to use for the critical values

    Returns
    -------
    T_d_crit: float
        The critical T^2 value for the principal components
    T_e_crit: float
        The critical T^2 value for the minor components
    Q_d_crit: float
        The critical Q value for the principal components
    Q_e_crit: float
        The critical Q value for the minor components
    """
    if alpha > 1 or alpha < 0:
        raise ValueError("Confidence level should be between 0 and 1")
    p = 1 - alpha

    n, m = X.shape

    P_d = P[:, :Md]
    P_e = P[:, Md:]

    scores_d = X @ P_d
    scores_e = X @ P_e

    E_d = X - scores_d @ P_d.T
    E_e = X - scores_e @ P_e.T

    SPEd = np.zeros((n,))
    SPEe = np.zeros((n,))
    for i in range(n):
        SPEd[i] = E_d[i, :] @ E_d[i, :].T
        SPEe[i] = E_e[i, :] @ E_e[i, :].T

    mu_d = np.mean(SPEd)
    nu_d = np.var(SPEd)

    mu_e = np.mean(SPEe)
    nu_e = np.var(SPEe)

    Q_d_crit = (nu_d / 2 / mu_d) * stats.chi2.ppf(p, 2 * mu_d ** 2 / nu_d)
    Q_e_crit = (nu_e / 2 / mu_e) * stats.chi2.ppf(p, 2 * mu_e ** 2 / nu_e)

    T_d_crit = (n - 1) * Md / (n - Md) * stats.f.ppf(p, Md, n - Md)
    T_e_crit = (n - 1) * Me / (n - Me) * stats.f.ppf(p, Me, n - Me)

    return(T_d_crit, T_e_crit, Q_d_crit, Q_e_crit)


def calculate_test_stats(Y: np.ndarray,
                         Md: int,
                         Omega_inv: np.ndarray = None) -> Tuple[np.ndarray,
                                                                np.ndarray,
                                                                np.ndarray,
                                                                np.ndarray]:
    """ Calculate T^2 and S^2 test statistics for slowest and fastest features

    Parameters
    ----------
    Y: np.ndarray
        An [m X n] matrix of features to calculate the statistics for
    Md: int
        The number of features considered to be 'slow', 1 <= Md <= m
    Omega_inv: np.ndarray
        An [m X m] diagonal matrix containing the speeds of the features, if
        nothing is passed, then S^2 is not calculated

    Returns
    -------
    Td: np.ndarray
        The T^2 values for the slowest features in an [n] array
    Te: np.ndarray
        The T^2 values for the fastest features in an [n] array
    Sd: np.ndarray
        The S^2 values for the slowest features in an [n - 1] array
    Se: np.ndarray
        The S^2 values for the fastest features in an [n - 1] array
    """
    n_test = Y.shape[1]
    Ydot = Y[:, 1:] - Y[:, :-1]

    Td = np.zeros((n_test,))
    Te = np.zeros((n_test,))
    Sd = np.zeros((n_test - 1,))
    Se = np.zeros((n_test - 1,))

    for i in range(n_test):
        Td[i] = Y[:Md, i].T @ Y[:Md, i]
        Te[i] = Y[Md:, i].T @ Y[Md:, i]
        if Omega_inv is None:
            continue  # Skip calculating S^2
        if i == n_test - 1:
            # Skip final sample for S^2
            continue
        Sd[i] = Ydot[:Md, i].T @ Omega_inv[:Md, :Md] @ Ydot[:Md, i]
        Se[i] = Ydot[Md:, i].T @ Omega_inv[Md:, Md:] @ Ydot[Md:, i]
    return(Td, Te, Sd, Se)


def calculate_test_stats_pca(X: np.ndarray, P: np.ndarray,
                             gamma_inv_d: np.ndarray, gamma_inv_e: np.ndarray,
                             Md: int) -> Tuple[np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray]:
    """ Calculate T^2 and S^2 test statistics for slowest and fastest features

    Parameters
    ----------
    X: np.ndarray
        An [n X m] data matrix to calculate the statistics for
    P: np.ndarray
        An [m X m] matrix of components
    Md: int
        The number of components to keep, 1 <= Md <= m

    Returns
    -------
    Td: np.ndarray
        The T^2 values for the principal components in an [n] array
    Te: np.ndarray
        The T^2 values for the minor components in an [n] array
    SPEd: np.ndarray
        The SPE^2 values for the principal components in an [n] array
    SPEe: np.ndarray
        The SPE^2 values for the minor components in an [n] array
    """
    n, m = X.shape

    P_d = P[:, :Md]
    P_e = P[:, Md:]

    scores_d = X @ P_d
    scores_e = X @ P_e

    E_d = X - scores_d @ P_d.T
    E_e = X - scores_e @ P_e.T

    Td = np.zeros((n,))
    Te = np.zeros((n,))
    SPEd = np.zeros((n,))
    SPEe = np.zeros((n,))

    for i in range(n):
        Td[i] = scores_d[i, :] @ gamma_inv_d @ scores_d[i, :].T
        Te[i] = scores_e[i, :] @ gamma_inv_e @ scores_e[i, :].T
        SPEd[i] = E_d[i, :] @ E_d[i, :].T
        SPEe[i] = E_e[i, :] @ E_e[i, :].T
    return(Td, Te, SPEd, SPEe)


def plot_test_stats(fig_name: str,
                    title: str,
                    Td_values: np.ndarray,
                    Te_values: np.ndarray,
                    Sd_values: np.ndarray,
                    Se_values: np.ndarray,
                    Td_crit: float,
                    Te_crit: float,
                    Sd_crit: float,
                    Se_crit: float,
                    show: bool = False,
                    save: bool = True) -> None:
    """ Plot the T^2 and S^2 test statistics for the slowest and fastest
    features

    Parameters
    ----------
    fig_name: str
        Name of figure and name of file to use if figure is saved
    title: str
        The title that shows up at the top of the figure
    Td_values: np.ndarray
        The T^2 values for the slowest features
    Te_values: np.ndarray
        The T^2 values for the fastest features
    Sd_values: np.ndarray
        The S^2 values for the slowest features
    Se_values: np.ndarray
        The S^2 values for the fastest features
    Td_crit: float
        The critical T^2 value for the slowest features
    Te_crit: float
        The critical T^2 value for the fastest features
    Sd_crit: float
        The critical S^2 value for the slowest features
    Se_crit: float
        The critical S^2 value for the fastest features
    show: bool
        If true, the figure will be shown while the program is running
    save: bool
        If true, the figure will be saved to the disk
    """
    _f, axs2d = plt.subplots(nrows=4, ncols=1, sharex=True)
    _f.set_size_inches(8, 6)

    Td_plot = axs2d[0]
    Td_plot.set_title(title)
    Td_plot.set_ylabel("$T^2_d$")
    Td_plot.plot(Td_values)
    Td_plot.plot([Td_crit] * len(Td_values))

    Te_plot = axs2d[1]
    Te_plot.set_ylabel("$T^2_e$")
    Te_plot.plot(Te_values)
    Te_plot.plot([Te_crit] * len(Te_values))

    Sd_plot = axs2d[2]
    Sd_plot.set_ylabel("$S^2_d$")
    Sd_plot.plot(Sd_values)
    Sd_plot.plot([Sd_crit] * len(Se_values))

    bot_plot = axs2d[3]
    bot_plot.set_ylabel("$S^2_e$")
    bot_plot.plot(Se_values)
    bot_plot.plot([Se_crit] * len(Se_values))
    bot_plot.set_xlabel("Sample")

    if show:
        plt.show()
    if save:
        plt.savefig(fig_name, dpi=350)
    plt.close(fig=_f)
    _f = None


def calculate_fault_contributions(X: np.ndarray,
                                  W: np.ndarray,
                                  Omega_inv: np.ndarray,
                                  Md: int,
                                  index: str = 'CDC') -> Tuple[np.ndarray,
                                                               np.ndarray,
                                                               np.ndarray,
                                                               np.ndarray]:
    """ Calculate the contributions of input variables to T^2 and S^2
    test statistics for slowest and fastest features

    Parameters
    ----------
    X: np.ndarray
        An [m X n] matrix of centered data to calculate the statistics for
    W: np.ndarray
        A [J X m] data to feature transformation matrix
    Omega_inv: np.ndarray
        An [m X m] diagonal matrix containing the speeds of the features
    Md: int
        The number of features considered to be 'slow', 1 <= Md <= m
    index: str
        The type of fault contribution to calculate, options are 'CDC', 'PDC',
        'DC', 'RBC', 'rCDC', 'rPDC', 'rDC', 'rRBC'.

    Returns
    -------
    Td_cont: np.ndarray
        The T^2 contributions for the slowest features in an [m X n] array
    Te_cont: np.ndarray
        The T^2 contributions for the fastest features in an [m X n] array
    Sd_cont: np.ndarray
        The S^2 contributions for the slowest features in an [m X n-1] array
    Se_cont: np.ndarray
        The S^2 contributions for the fastest features in an [m X n-1] array
    """
    X_dot = X[:, 1:] - X[:, :-1]
    cov_X = np.cov(X)
    cov_X_dot = np.cov(X_dot)

    W_d = W[:Md, :]
    W_e = W[Md:, :]

    Omega_inv_d = Omega_inv[:Md, :Md]
    Omega_inv_e = Omega_inv[Md:, Md:]

    M_Td = W_d.T @ W_d
    M_Te = W_e.T @ W_e
    M_Sd = W_d.T @ Omega_inv_d @ W_d
    M_Se = W_e.T @ Omega_inv_e @ W_e

    Td_fd = fd.GenericFaultDiagnosisModel(M_Td, cov_X)
    Te_fd = fd.GenericFaultDiagnosisModel(M_Te, cov_X)
    Sd_fd = fd.GenericFaultDiagnosisModel(M_Sd, cov_X_dot)
    Se_fd = fd.GenericFaultDiagnosisModel(M_Se, cov_X_dot)

    m, n_test = X.shape
    Td_cont = np.zeros((m, n_test))
    Te_cont = np.zeros((m, n_test,))
    Sd_cont = np.zeros((m, n_test - 1,))
    Se_cont = np.zeros((m, n_test - 1,))

    for i in range(n_test):
        Td_cont[:, i] = Td_fd.get_contributions(X[:, i], index)[index]
        Te_cont[:, i] = Te_fd.get_contributions(X[:, i], index)[index]
        if i == n_test - 1:
            # Skip final sample for S^2
            continue
        Sd_cont[:, i] = Sd_fd.get_contributions(X_dot[:, i], index)[index]
        Se_cont[:, i] = Se_fd.get_contributions(X_dot[:, i], index)[index]
    return(Td_cont, Te_cont, Sd_cont, Se_cont)


def calculate_fault_contributions_pca(X: np.ndarray,
                                      P: np.ndarray,
                                      gamma_inv_d: np.ndarray,
                                      gamma_inv_e: np.ndarray,
                                      Md: int,
                                      index: str = 'CDC') -> Tuple[np.ndarray,
                                                                   np.ndarray,
                                                                   np.ndarray,
                                                                   np.ndarray]:
    """ Calculate the contributions of input variables to T^2 and S^2
    test statistics for slowest and fastest features

    Parameters
    ----------
    X: np.ndarray
        An [m X n] matrix of centered data to calculate the statistics for
    W: np.ndarray
        A [J X m] data to feature transformation matrix
    Omega_inv: np.ndarray
        An [m X m] diagonal matrix containing the speeds of the features
    Md: int
        The number of features considered to be 'slow', 1 <= Md <= m
    index: str
        The type of fault contribution to calculate, options are 'CDC', 'PDC',
        'DC', 'RBC', 'rCDC', 'rPDC', 'rDC', 'rRBC'.

    Returns
    -------
    Td_cont: np.ndarray
        The T^2 contributions for the slowest features in an [m X n] array
    Te_cont: np.ndarray
        The T^2 contributions for the fastest features in an [m X n] array
    Sd_cont: np.ndarray
        The S^2 contributions for the slowest features in an [m X n-1] array
    Se_cont: np.ndarray
        The S^2 contributions for the fastest features in an [m X n-1] array
    """
    n_test, m = X.shape
    if index.lower() != "cdc":
        raise NotImplementedError

    P_d = P[:, :Md]
    P_e = P[:, Md:]

    M_Td = P_d @ gamma_inv_d @ P_d.T
    M_Te = P_e @ gamma_inv_e @ P_e.T
    M_SPEd = (np.eye(m) - P_d @ P_d.T) @ (np.eye(m) - P_d @ P_d.T).T
    M_SPEe = (np.eye(m) - P_e @ P_e.T) @ (np.eye(m) - P_e @ P_e.T).T

    Td_fd = fd.GenericFaultDiagnosisModel(M_Td, np.eye(m))
    Te_fd = fd.GenericFaultDiagnosisModel(M_Te, np.eye(m))
    SPEd_fd = fd.GenericFaultDiagnosisModel(M_SPEd, np.eye(m))
    SPEe_fd = fd.GenericFaultDiagnosisModel(M_SPEe, np.eye(m))

    Td_cont = np.zeros((m, n_test))
    Te_cont = np.zeros((m, n_test,))
    SPEd_cont = np.zeros((m, n_test,))
    SPEe_cont = np.zeros((m, n_test,))

    for i in range(n_test):
        Td_cont[:, i] = Td_fd.get_contributions(X[i, :], index)[index]
        Te_cont[:, i] = Te_fd.get_contributions(X[i, :], index)[index]
        SPEd_cont[:, i] = SPEd_fd.get_contributions(X[i, :], index)[index]
        SPEe_cont[:, i] = SPEe_fd.get_contributions(X[i, :], index)[index]
    return(Td_cont, Te_cont, SPEd_cont, SPEe_cont)
