from src.sfamanopt import incmssfa
from SlowFeatureAnalysis.src.sfafd import incsfa, rsfa
import src.sfamanopt.fault_diagnosis as fd

import numpy as np
import matplotlib.pyplot as plt

import tepimport

if __name__ == "__main__":
    alpha = 0.01
    Md = [64, 99, 89, 86]
    lagged_samples = 2
    idv = range(22)
    """Import Data"""
    ignored_var = list(range(22, 41))
    X = tepimport.import_sets([0], skip_test=True)[0][1]
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
        Y_rsfa = (W_rsfa.T @ X_test)
        Y_incsfa = (W_incsfa.T @ X_test)
        Y_incsfa_svd = (W_incsfa_svd.T @ X_test)
        Y_incmssfa = (W_incmssfa.T @ X_test)

        stats_rsfa = fd.calculate_test_stats(Y_rsfa, Md[0], Omega_inv_rsfa)
        stats_incsfa = fd.calculate_test_stats(Y_incsfa, Md[1],
                                               Omega_inv_incsfa)
        stats_incsfa_svd = fd.calculate_test_stats(Y_incsfa_svd, Md[2],
                                                   Omega_inv_incsfa_svd)
        stats_incmssfa = fd.calculate_test_stats(Y_incmssfa, Md[3],
                                                 Omega_inv_incmssfa)

        results.append((name, stats_rsfa, stats_incsfa,
                        stats_incsfa_svd, stats_incmssfa))

    fdr_results = []
    for (name, stats_rsfa, stats_incsfa,
         stats_incsfa_svd, stats_incmssfa) in results:
        T_fault_index = 160 - lagged_samples
        S_fault_index = 159 - lagged_samples

        idv_results = [name]
        stats = zip(["RSFA", "IncSFA", "IncSFA SVD", "IncMSSFA"],
                    [stats_rsfa, stats_incsfa,
                     stats_incsfa_svd, stats_incmssfa])
        for i, (method, (Td, Te, Sd, Se)) in enumerate(stats):
            Td_crit, _, _, _ = fd.calculate_crit_values(n, Md[i], Me[i], alpha)
            if name == "IDV(0)":
                FDR = 0
                FAR = np.count_nonzero(Td > Td_crit) / len(Td)
            else:
                FDR = (np.count_nonzero(Td[T_fault_index:] > Td_crit)
                       / (len(Td) - T_fault_index))
                FAR = (np.count_nonzero(Td[:T_fault_index] > Td_crit)
                       / T_fault_index)
            idv_results.append(FDR)
            idv_results.append(FAR)
        fdr_results.append(idv_results)

    latex_text = ""
    for (name, fdr_rsfa, far_rsfa,
         fdr_incsfa, far_incsfa,
         fdr_incsfa_svd, far_incsfa_svd,
         fdr_incmssfa, far_incmssfa) in fdr_results:
        latex_text += f"{name} & "

        fdr_array = [fdr_rsfa, fdr_incsfa, fdr_incsfa_svd, fdr_incmssfa]
        far_array = [far_rsfa, far_incsfa, far_incsfa_svd, far_incmssfa]
        fdr_array = np.around(np.asarray(fdr_array), 3)
        far_array = np.around(np.asarray(far_array), 3)

        fdr_order = np.argsort(-1 * fdr_array)
        far_order = np.argsort(far_array)

        bolded_fdr = -1
        bolded_far = -1
        if not np.isclose(fdr_array[fdr_order[0]], fdr_array[fdr_order[1]]):
            bolded_fdr = fdr_order[0]
        if not np.isclose(far_array[far_order[0]], far_array[far_order[1]]):
            bolded_far = far_order[0]

        for i, (fdr, far) in enumerate(zip(fdr_array, far_array)):
            if i == bolded_fdr:
                latex_text += "\\textbf{" + f"{fdr:.3f}" + "} & "
            else:
                latex_text += f"{fdr:.3f} & "

            if i == bolded_far:
                latex_text += "\\textbf{" + f"{far:.3f}" + "} & "
            else:
                latex_text += f"{far:.3f} & "
        latex_text = f"{latex_text[:-3]} \\\\ \n"
    print(latex_text)
