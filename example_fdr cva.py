import src.sfamanopt.mssfa as mssfa
import src.sfamanopt.ssfa as ssfa
import src.sfamanopt.sfa as sfa
import src.sfamanopt.fault_diagnosis as fd
import src.sfamanopt.load_cva as cva

import numpy as np
from sklearn.decomposition import SparsePCA

if __name__ == "__main__":
    alpha = 0.01
    Md = 12
    """Import Data"""
    data_sets = cva.import_sets()
    X = np.hstack((data_sets[0][1][0][0],
                   data_sets[0][1][1][0],
                   data_sets[0][1][2][0]))

    tests = []
    for name, data, f_range in data_sets[1:]:  # skip training set
        for i, single_set in enumerate(data):
            tests.append((f"{name}.{i+1}", single_set[0], f_range[i]))

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
    # W_ssfa, Omega_inv_ssfa, _, _, _ = ssfa_object.run(X, Md)
    W_ssfa, Omega_inv_ssfa = W_sfa, Omega_inv_sfa
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
    for name, test_data, fault_samples in tests:
        X_test = ((test_data - X_mean) / X_std)

        Y_sfa = (W_sfa.T @ X_test)
        Y_ssfa = (W_ssfa.T @ X_test)
        Y_spca = spca.transform(X_test.T).T
        Y_mssfa = (W_mssfa.T @ X_test)

        stats_sfa = fd.calculate_test_stats(Y_sfa, Md, Omega_inv_sfa)

        stats_ssfa = fd.calculate_test_stats(Y_ssfa, Md, Omega_inv_ssfa)
        for i in range(X_test.shape[1]):
            stats_ssfa[0][i] = (Y_ssfa[:Md, i].T @ Lambda_inv_ssfa[:Md, :Md]
                                @ Y_ssfa[:Md, i])
            stats_ssfa[1][i] = (Y_ssfa[Md:, i].T @ Lambda_inv_ssfa[Md:, Md:]
                                @ Y_ssfa[Md:, i])

        stats_spca = fd.calculate_test_stats_pca(X_test.T, P, gamma_inv_d,
                                                 gamma_inv_e, Md)

        stats_mssfa = fd.calculate_test_stats(Y_mssfa, Md, Omega_inv_mssfa)

        results.append((name, fault_samples,
                        stats_sfa, stats_ssfa,
                        stats_spca, stats_mssfa))

    Tdc, Tec, Sdc, Sec = fd.calculate_crit_values(n, Md, Me, alpha)
    Tdc_pca, Tec_pca, SPEd, SPEe = fd.calculate_crit_values_pca(X.T, P, n, Md,
                                                                Me, alpha)

    fdr_results = []
    for name, fault_samples, s_sfa, s_ssfa, s_spca, s_mssfa in results:
        cva_results = [name]
        stats = zip(["SFA", "SSFA", "SPCA", "MSSFA"],
                    [s_sfa, s_ssfa, s_spca, s_mssfa])
        for method, (Td, Te, Sd, Se) in stats:
            if method == "SPCA":
                crit_val = Tdc_pca
            else:
                crit_val = Tdc
            f_start = fault_samples[0]
            f_stop = fault_samples[-1]
            n_faults = f_stop - f_start + 1

            FDR = (np.count_nonzero(Td[f_start:f_stop+1] > crit_val)
                   / n_faults)

            false_alarms = np.count_nonzero(Td[:f_start] > crit_val)
            if f_stop < len(Td):  # stopped fault with more data after
                false_alarms += np.count_nonzero(Td[f_stop+1:] > crit_val)
            FAR = false_alarms / (len(Td) - n_faults)
            cva_results.append(FDR)
            cva_results.append(FAR)
        fdr_results.append(cva_results)

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
