import src.sfamanopt.mssfa as mssfa
import src.sfamanopt.ssfa as ssfa
import src.sfamanopt.sfa as methods

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import SparsePCA

import tepimport

if __name__ == "__main__":
    load_ssfa = False
    alpha = 0.01
    Md = [55, 74, 48, 85]
    lagged_samples = 2
    thresholds = [10 ** (-x) for x in range(13)]
    thresholds.append(0)
    """Import Data"""
    X = tepimport.import_sets((0), skip_test=True)[0]

    ignored_var = list(range(22, 41))
    X = np.delete(X[1], ignored_var, axis=0)

    X = tepimport.add_lagged_samples(X, lagged_samples)

    m = X.shape[0]
    n = X.shape[1]
    X_mean = np.mean(X, axis=1).reshape((-1, 1))
    X = X - X_mean
    X_std = np.std(X, axis=1).reshape((-1, 1))
    X = X / X_std

    """Train Models"""
    sfa_object = methods.SFA()
    W_sfa, Omega_inv_sfa = sfa_object.run(X, Md[0])

    ssfa_object = ssfa.SSFA()
    if load_ssfa:
        with open('ssfa_matrix.npy', 'rb') as f:
            W_ssfa = np.load(f)
            Omega_inv_ssfa = np.load(f)
    else:
        W_ssfa, Omega_inv_ssfa, _, _, _ = ssfa_object.run(X, Md[1])
    Lambda_inv_ssfa = np.linalg.pinv(W_ssfa.T @ W_ssfa)

    spca = SparsePCA(n_components=Md[2], max_iter=500, tol=1e-6)
    T = spca.fit_transform(X.T)
    P_spca = spca.components_.T
    print(f"SPCA converged in {spca.n_iter_} iterations")
    Lambda_spca = np.cov(T.T)
    Lambda_inv_spca = np.diag(np.diag(Lambda_spca) ** -1)

    mssfa_object = mssfa.MSSFA("chol", "l1")
    W_mssfa, Omega_inv_mssfa, _, _, _ = mssfa_object.run(X, Md[3])

    """Sparsity Calculation"""
    sfa_sparse = np.abs(W_sfa) / np.abs(W_sfa).sum(axis=0, keepdims=1)
    ssfa_sparse = np.abs(W_ssfa) / np.abs(W_ssfa).sum(axis=0, keepdims=1)
    spca_sparse = np.abs(P_spca) / np.abs(P_spca).sum(axis=0, keepdims=1)
    mssfa_sparse = np.abs(W_mssfa) / np.abs(W_mssfa).sum(axis=0, keepdims=1)

    """Print Slowest Feature"""
    index_labels = [
        "XMEAS(01) A Feed  (stream 1)",
        "XMEAS(02) D Feed  (stream 2)",
        "XMEAS(03) E Feed  (stream 3)",
        "XMEAS(04) A and C Feed  (stream 4)",
        "XMEAS(05) Recycle Flow  (stream 8)",
        "XMEAS(06) Reactor Feed Rate  (stream 6)",
        "XMEAS(07) Reactor Pressure",
        "XMEAS(08) Reactor Level",
        "XMEAS(09) Reactor Temperature",
        "XMEAS(10) Purge Rate (stream 9)",
        "XMEAS(11) Product Separator Temperature",
        "XMEAS(12) Product Separator Level",
        "XMEAS(13) Prod Separator Pressure",
        "XMEAS(14) Prod Separator Underflow (stream 10)",
        "XMEAS(15) Stripper Level",
        "XMEAS(16) Stripper Pressure",
        "XMEAS(17) Stripper Underflow (stream 11)",
        "XMEAS(18) Stripper Temperature",
        "XMEAS(19) Stripper Steam Flow",
        "XMEAS(20) Compressor Work",
        "XMEAS(21) Reactor CW Outlet Temperature",
        "XMEAS(22) Separator CW Outlet Temperature",
        "XMV(01) D Feed Flow (stream 2)",
        "XMV(02) E Feed Flow (stream 3)",
        "XMV(03) A Feed Flow (stream 1)",
        "XMV(04) A and C Feed Flow (stream 4)",
        "XMV(05) Compressor Recycle Valve",
        "XMV(06) Purge Valve (stream 9)",
        "XMV(07) Separator Pot Liquid Flow (stream 10)",
        "XMV(08) Stripper Liquid Product Flow (stream 11)",
        "XMV(09) Stripper Steam Valve",
        "XMV(10) Reactor CW Flow",
        "XMV(11) Condenser CW Flow",
    ]
    lagged_labels = []
    for i in range(1, lagged_samples + 1):
        for lbl in index_labels:
            lagged_labels.append(f"{lbl} (t - {i})")
    index_labels += lagged_labels

    sfa_slowest = np.argsort(-1 * np.abs(sfa_sparse[:, 0]))
    ssfa_slowest = np.argsort(-1 * np.abs(ssfa_sparse[:, 0]))
    spca_slowest = np.argsort(-1 * np.abs(spca_sparse[:, 0]))
    mssfa_slowest = np.argsort(-1 * np.abs(mssfa_sparse[:, 0]))
    print("SFA")
    for i in range(10):
        print(f"{index_labels[sfa_slowest[i]]}, "
              f"{sfa_sparse[:, 0][sfa_slowest[i]] * 100:.2f}")
    remainder = 1 - np.sum(sfa_sparse[:, 0][sfa_slowest[:10]])
    print(f"Others: {remainder * 100:.2f}")
    print("Sparse SFA")
    for i in range(10):
        print(f"{index_labels[ssfa_slowest[i]]}, "
              f"{ssfa_sparse[:, 0][ssfa_slowest[i]] * 100:.2f}")
    remainder = 1 - np.sum(ssfa_sparse[:, 0][ssfa_slowest[:10]])
    print(f"Others: {remainder * 100:.2f}")
    print("Sparse PCA")
    for i in range(10):
        print(f"{index_labels[spca_slowest[i]]}, "
              f"{spca_sparse[:, 0][spca_slowest[i]] * 100:.2f}")
    remainder = 1 - np.sum(spca_sparse[:, 0][spca_slowest[:10]])
    print(f"Others: {remainder * 100:.2f}")
    print("Manifold Sparse SFA")
    for i in range(10):
        print(f"{index_labels[mssfa_slowest[i]]}, "
              f"{mssfa_sparse[:, 0][mssfa_slowest[i]] * 100:.2f}")
    remainder = 1 - np.sum(mssfa_sparse[:, 0][mssfa_slowest[:10]])
    print(f"Others: {remainder * 100:.2f}")

    """Sparsity Plot"""
    for threshold in thresholds:
        _f_sparse, axim = plt.subplots(nrows=1, ncols=4, sharey=True)
        _f_sparse.set_size_inches(21, 9)

        im_sfa = np.abs(np.copy(sfa_sparse))
        im_ssfa = np.abs(np.copy(ssfa_sparse))
        im_spca = np.abs(np.copy(spca_sparse))
        im_mssfa = np.abs(np.copy(mssfa_sparse))

        im_sfa[im_sfa <= threshold] = np.NaN
        im_ssfa[im_ssfa <= threshold] = np.NaN
        im_spca[im_spca <= threshold] = np.NaN
        im_mssfa[im_mssfa <= threshold] = np.NaN

        sfa_S = np.count_nonzero(np.isnan(im_sfa)) / np.size(im_sfa)
        ssfa_S = np.count_nonzero(np.isnan(im_ssfa)) / np.size(im_ssfa)
        spca_S = np.count_nonzero(np.isnan(im_spca)) / np.size(im_spca)
        mssfa_S = np.count_nonzero(np.isnan(im_mssfa))/np.size(im_mssfa)
        axim[0].set_title(f"SFA ({sfa_S:.3f})")
        axim[1].set_title(f"Sparse SFA ({ssfa_S:.3f})")
        axim[2].set_title(f"Sparse PCA ({spca_S:.3f})")
        axim[3].set_title(f"Manifold Sparse SFA ({mssfa_S:.3f})")

        axim[0].imshow(np.abs(im_sfa))
        axim[1].imshow(np.abs(im_ssfa))
        axim[2].imshow(np.abs(im_spca))
        im = axim[3].imshow(np.abs(im_mssfa))

        axim[0].set_ylabel("Input Signal")
        axim[0].set_xlabel("Feature")
        axim[1].set_xlabel("Feature")
        axim[2].set_xlabel("Component")
        axim[3].set_xlabel("Feature")

        _f_sparse.colorbar(im, ax=axim, aspect=60)

        if threshold == 0:
            magnitude = 0
            _f_sparse.text(0.2, 0.95, f'J = {Md} | Threshold = 0', fontsize=24)
        else:
            magnitude = int(np.abs(np.floor(np.log10(threshold))))
            _f_sparse.text(0.2, 0.95, f'J = {Md} | Threshold = 1e-{magnitude}',
                           fontsize=24)
        plt.savefig(f"plots/Sparsity/Sparsity_comparison_{magnitude}.png",
                    dpi=350)
        plt.close(fig=_f_sparse)
        _f_sparse = None
