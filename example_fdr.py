import src.sfamanopt.mssfa as mssfa
import src.sfamanopt.ssfa as ssfa
import src.sfamanopt.sfa as sfa
import src.sfamanopt.fault_diagnosis as fd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import SparsePCA

import tepimport

if __name__ == "__main__":
    alpha = 0.01
    Md = 55
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
    Me = m - Md

    """Train Model"""
    sfa_object = sfa.SFA()
    W_sfa, Omega_inv_sfa = sfa_object.run(X, Md)

    ssfa_object = ssfa.SSFA()
    W_ssfa, Omega_inv_ssfa, _, _, _ = ssfa_object.run(X, Md)
    Lambda_inv_ssfa = np.linalg.pinv(W_ssfa.T @ W_ssfa)

    spca = SparsePCA(n_components=Md, max_iter=500, tol=1e-6)
    T = spca.fit_transform(X.T)
    print(f"SPCA converged in {spca.n_iter_} iterations")
    Lambda_spca = np.cov(T.T)
    Lambda_inv_spca = np.diag(np.diag(Lambda_spca) ** -1)

    mssfa_object = mssfa.MSSFA("chol", "l1")
    W_mssfa, Omega_inv_mssfa, _, _, _ = mssfa_object.run(X, Md)

    results = []
    for name, test_data in tests:
        X_test = ((test_data - X_mean) / X_std)
        Y_sfa = (W_sfa.T @ X_test)
        Y_ssfa = (W_ssfa.T @ X_test)
        Y_spca = spca.transform(X_test.T).T
        Y_mssfa = (W_mssfa.T @ X_test)

        stats_sfa = fd.calculate_test_stats(Y_sfa, Md, Omega_inv_sfa)

        stats_ssfa = fd.calculate_test_stats(Y_ssfa, Md, Omega_inv_ssfa)
        for i in range(n_test):
            stats_ssfa[0][i] = (Y_ssfa[:Md, i].T @ Lambda_inv_ssfa[:Md, :Md]
                                @ Y_ssfa[:Md, i])
            stats_ssfa[1][i] = (Y_ssfa[Md:, i].T @ Lambda_inv_ssfa[Md:, Md:]
                                @ Y_ssfa[Md:, i])

        stats_spca = fd.calculate_test_stats_pca(X_test, Md,
                                                 spca.components_.T,
                                                 Lambda_inv_spca)

        stats_mssfa = fd.calculate_test_stats(Y_mssfa, Md, Omega_inv_mssfa)

        results.append((name, stats_sfa, stats_ssfa, stats_spca, stats_mssfa))

    Tdc, Tec, Sdc, Sec = fd.calculate_crit_values(n_test, Md, Me, alpha)

    fdr_results = []
    for name, stats_sfa, stats_ssfa, stats_spca, stats_mssfa in results:
        T_fault_index = 160 - lagged_samples
        S_fault_index = 159 - lagged_samples

        idv_results = [name]
        for Td, Te, Sd, Se in [stats_sfa, stats_ssfa, stats_spca, stats_mssfa]:
            if name == "IDV(0)":
                FDR = 0
                FAR = np.count_nonzero(Td > Tdc) / len(Td)
            else:
                FDR = (np.count_nonzero(Td[T_fault_index:] > Tdc)
                       / (len(Td) - T_fault_index))
                FAR = (np.count_nonzero(Td[:T_fault_index] > Tdc)
                       / T_fault_index)
            idv_results.append(FDR)
            idv_results.append(FAR)
        fdr_results.append(idv_results)

    latex_text = ""
    for (name, fdr_sfa, far_sfa,
         fdr_ssfa, far_ssfa,
         fdr_spca, far_spca,
         fdr_mssfa, far_mssfa) in fdr_results:
        latex_text += f"{name} & "

        fdr_array = [fdr_sfa, fdr_ssfa, fdr_spca, fdr_mssfa]
        far_array = [far_sfa, far_ssfa, far_spca, far_mssfa]
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
