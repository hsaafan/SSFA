import numpy as np
import matplotlib.pyplot as plt

import tepimport

from src.sfamanopt import incmssfa
from SlowFeatureAnalysis.src.sfafd import incsfa, rsfa

if __name__ == "__main__":
    alpha = 0.01
    Md = 55
    lagged_samples = 2
    threshold = 1e-12
    """Import Data"""
    X = tepimport.import_sets((0), skip_test=True)[0]

    ignored_var = list(range(22, 41))
    X = np.delete(X[1], ignored_var, axis=0)

    X = tepimport.add_lagged_samples(X, lagged_samples)
    X_mean = np.mean(X, axis=1).reshape((-1, 1))
    X = X - X_mean
    X_std = np.std(X, axis=1).reshape((-1, 1))
    X = X / X_std

    m, n = X.shape

    """Train Models"""
    rsfa_object = rsfa.RSFA(m, Md, Md, L=2, conv_tol=0)
    incsfa_object = incsfa.IncSFA(m, Md, Md, L=2, conv_tol=0)
    incsfa_svd_object = incsfa.IncSFA(m, Md, Md, L=2, conv_tol=0)
    incmssfa_object = incmssfa.IncMSSFA()

    for i in range(n):
        rsfa_object.add_data(X[:, i])

    for i in range(n):
        incsfa_object.add_data(X[:, i], update_monitors=False,
                               use_svd_whitening=False)

    for i in range(n):
        incsfa_svd_object.add_data(X[:, i], update_monitors=False,
                                   use_svd_whitening=True)

    W_rsfa = (rsfa_object.standardization_node.whitening_matrix
              @ rsfa_object.transformation_matrix)
    W_incsfa = (incsfa_object.standardization_node.whitening_matrix
                @ incsfa_object.transformation_matrix)
    W_incsfa_svd = (incsfa_svd_object.standardization_node.whitening_matrix
                    @ incsfa_svd_object.transformation_matrix)
    W_incmssfa, _, sparsity_evo = incmssfa_object.run(X, Md, L=2,
                                                      calculate_sparsity=True)

    """Sparsity Calculation"""
    rsfa_sparse = np.abs(W_rsfa) / np.abs(W_rsfa).sum(axis=0, keepdims=1)
    incsfa_sparse = np.abs(W_incsfa) / np.abs(W_incsfa).sum(axis=0, keepdims=1)
    incsfa_svd_sparse = (np.abs(W_incsfa_svd)
                         / np.abs(W_incsfa_svd).sum(axis=0, keepdims=1))
    incmssfa_sparse = np.abs(W_incmssfa) / np.abs(W_incmssfa).sum(axis=0,
                                                                  keepdims=1)

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

    """Slowest Feature"""
    rsfa_slowest = np.argsort(-1 * np.abs(rsfa_sparse[:, 0]))
    incsfa_slowest = np.argsort(-1 * np.abs(incsfa_sparse[:, 0]))
    incsfa_svd_slowest = np.argsort(-1 * np.abs(incsfa_svd_sparse[:, 0]))
    incmssfa_slowest = np.argsort(-1 * np.abs(incmssfa_sparse[:, 0]))
    print("RSFA")
    for i in range(10):
        print(f"{index_labels[rsfa_slowest[i]]}, "
              f"{rsfa_sparse[:, 0][rsfa_slowest[i]] * 100:.2f}")
    remainder = 1 - np.sum(rsfa_sparse[:, 0][rsfa_slowest[:10]])
    print(f"Others: {remainder * 100:.2f}")
    print("IncSFA")
    for i in range(10):
        print(f"{index_labels[incsfa_slowest[i]]}, "
              f"{incsfa_sparse[:, 0][incsfa_slowest[i]] * 100:.2f}")
    remainder = 1 - np.sum(incsfa_sparse[:, 0][incsfa_slowest[:10]])
    print(f"Others: {remainder * 100:.2f}")
    print("IncSFA SVD")
    for i in range(10):
        print(f"{index_labels[incsfa_svd_slowest[i]]}, "
              f"{incsfa_svd_sparse[:, 0][incsfa_svd_slowest[i]] * 100:.2f}")
    remainder = 1 - np.sum(incsfa_svd_sparse[:, 0][incsfa_svd_slowest[:10]])
    print(f"Others: {remainder * 100:.2f}")
    print("IncMSSFA")
    for i in range(10):
        print(f"{index_labels[incmssfa_slowest[i]]}, "
              f"{incmssfa_sparse[:, 0][incmssfa_slowest[i]] * 100:.2f}")
    remainder = 1 - np.sum(incmssfa_sparse[:, 0][incmssfa_slowest[:10]])
    print(f"Others: {remainder * 100:.2f}")

    """Sparsity Plot"""
    _f_sparse, axim = plt.subplots(nrows=1, ncols=4, sharey=True)
    _f_sparse.set_size_inches(21, 9)

    im_rsfa = np.abs(np.copy(rsfa_sparse))
    im_incsfa = np.abs(np.copy(incsfa_sparse))
    im_incsfa_svd = np.abs(np.copy(incsfa_svd_sparse))
    im_incmssfa = np.abs(np.copy(incmssfa_sparse))

    im_rsfa[im_rsfa <= threshold] = np.NaN
    im_incsfa[im_incsfa <= threshold] = np.NaN
    im_incsfa_svd[im_incsfa_svd <= threshold] = np.NaN
    im_incmssfa[im_incmssfa <= threshold] = np.NaN

    rsfa_S = np.count_nonzero(np.isnan(im_rsfa)) / np.size(im_rsfa)
    incsfa_S = np.count_nonzero(np.isnan(im_incsfa)) / np.size(im_incsfa)
    incsfa_svd_S = (np.count_nonzero(np.isnan(im_incsfa_svd))
                    / np.size(im_incsfa_svd))
    incmssfa_S = np.count_nonzero(np.isnan(im_incmssfa))/np.size(im_incmssfa)
    axim[0].set_title(f"RSFA ({rsfa_S:.3f})", fontsize=20)
    axim[1].set_title(f"IncSFA ({incsfa_S:.3f})", fontsize=20)
    axim[2].set_title(f"IncSFA SVD ({incsfa_svd_S:.3f})", fontsize=20)
    axim[3].set_title(f"IncMSSFA ({incmssfa_S:.3f})", fontsize=20)

    axim[0].imshow(np.abs(im_rsfa))
    axim[1].imshow(np.abs(im_incsfa))
    axim[2].imshow(np.abs(im_incsfa_svd))
    im = axim[3].imshow(np.abs(im_incmssfa))

    axim[0].set_ylabel("Input Signal", fontsize=20)
    axim[0].set_xlabel("Feature", fontsize=20)
    axim[1].set_xlabel("Feature", fontsize=20)
    axim[2].set_xlabel("Feature", fontsize=20)
    axim[3].set_xlabel("Feature", fontsize=20)

    for i in range(4):
        axim[i].tick_params(axis='both', which='major', labelsize=15)

    cbar = _f_sparse.colorbar(im, ax=axim, aspect=60)
    cbar.ax.tick_params(labelsize=15)

    plt.savefig(f"plots/incremental_sparsity.png", dpi=350)
    plt.close(fig=_f_sparse)
    _f_sparse = None

    _f, ax = plt.subplots()

    ax.set_title(f"IncMSSFA Sparsity")

    ax.plot(sparsity_evo)

    ax.set_ylabel("Sparsity Fraction")
    ax.set_xlabel("Training Sample")

    plt.savefig(f"plots/incmssfa_sparsity.png", dpi=350)
    plt.close(fig=_f)
    _f = None
