import src.sfamanopt.mssfa as mssfa

import numpy as np
import matplotlib.pyplot as plt

import tepimport

if __name__ == "__main__":
    load_ssfa = False
    alpha = 0.01
    Md = 55
    lagged_samples = 2
    threshold = 1e-12
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
    mssfa_l1 = mssfa.MSSFA("chol", "l1")
    mssfa_l2 = mssfa.MSSFA("chol", "l2")
    mssfa_en = mssfa.MSSFA("chol", "elastic net")
    W_l1, Omega_inv_l1, _, _ = mssfa_l1.run(X, Md)
    W_l2, Omega_inv_l2, _, _ = mssfa_l2.run(X, Md)
    W_en, Omega_inv_en, _, _ = mssfa_en.run(X, Md)

    """Sparsity Calculation"""
    l1_sparse = np.abs(W_l1) / np.abs(W_l1).sum(axis=0, keepdims=1)
    l2_sparse = np.abs(W_l2) / np.abs(W_l2).sum(axis=0, keepdims=1)
    en_sparse = np.abs(W_en) / np.abs(W_en).sum(axis=0, keepdims=1)

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

    l1_slowest = np.argsort(-1 * np.abs(l1_sparse[:, 0]))
    l2_slowest = np.argsort(-1 * np.abs(l2_sparse[:, 0]))
    en_slowest = np.argsort(-1 * np.abs(en_sparse[:, 0]))
    print("Manifold Sparse SFA L1")
    for i in range(10):
        print(f"{index_labels[l1_slowest[i]]}, "
              f"{l1_sparse[:, 0][l1_slowest[i]] * 100:.2f}")
    remainder = 1 - np.sum(l1_sparse[:, 0][l1_slowest[:10]])
    print(f"Others: {remainder * 100:.2f}")
    print("Manifold Sparse SFA L2")
    for i in range(10):
        print(f"{index_labels[l2_slowest[i]]}, "
              f"{l2_sparse[:, 0][l2_slowest[i]] * 100:.2f}")
    remainder = 1 - np.sum(l2_sparse[:, 0][l2_slowest[:10]])
    print(f"Others: {remainder * 100:.2f}")
    print("Manifold Sparse SFA EN")
    for i in range(10):
        print(f"{index_labels[en_slowest[i]]}, "
              f"{en_sparse[:, 0][en_slowest[i]] * 100:.2f}")
    remainder = 1 - np.sum(en_sparse[:, 0][en_slowest[:10]])
    print(f"Others: {remainder * 100:.2f}")

    """Sparsity Plot"""
    _f_sparse, axim = plt.subplots(nrows=1, ncols=3, sharey=True)
    _f_sparse.set_size_inches(21, 9)

    im_l1 = np.abs(np.copy(l1_sparse))
    im_l2 = np.abs(np.copy(l2_sparse))
    im_en = np.abs(np.copy(en_sparse))

    im_l1[im_l1 <= threshold] = np.NaN
    im_l2[im_l2 <= threshold] = np.NaN
    im_en[im_en <= threshold] = np.NaN

    l1_S = np.count_nonzero(np.isnan(im_l1)) / np.size(im_l1)
    l2_S = np.count_nonzero(np.isnan(im_l2)) / np.size(im_l2)
    en_S = np.count_nonzero(np.isnan(im_en)) / np.size(im_en)

    axim[0].set_title(f"$l_1$ ({l1_S:.3f})", fontsize=20)
    axim[1].set_title(f"$l_2$ ({l2_S:.3f})", fontsize=20)
    axim[2].set_title(f"Elastic Net ({en_S:.3f})", fontsize=20)

    axim[0].imshow(np.abs(im_l1))
    axim[1].imshow(np.abs(im_l2))
    im = axim[2].imshow(np.abs(im_en))

    axim[0].set_ylabel("Input Signal", fontsize=20)
    axim[0].set_xlabel("Feature", fontsize=20)
    axim[1].set_xlabel("Feature", fontsize=20)
    axim[2].set_xlabel("Feature", fontsize=20)

    for i in range(3):
        axim[i].tick_params(axis='both', which='major', labelsize=15)

    cbar = _f_sparse.colorbar(im, ax=axim, aspect=60)
    cbar.ax.tick_params(labelsize=15)

    if threshold == 0:
        magnitude = 0
    else:
        magnitude = int(np.abs(np.floor(np.log10(threshold))))

    plt.savefig(f"plots/Sparsity/Sparsity_comparison_reg_{magnitude}.png",
                dpi=350)
    plt.close(fig=_f_sparse)
    _f_sparse = None
