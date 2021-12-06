import src.sfamanopt.ssfa as ssfa
import src.sfamanopt.paper_ssfa as oldssfa
import src.sfamanopt.fault_diagnosis as fd

import numpy as np
import matplotlib.pyplot as plt

import tepimport

if __name__ == "__main__":
    alpha = 0.01
    J = 55  # How many to calculate
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
    # Ours
    ssfa_object = ssfa.SSFA("chol", "l1")
    W, _, _, _ = ssfa_object.run(X, J)

    # Other sparse
    paper_ssfa_object = oldssfa.PaperSSFA()
    W_old, _, _, _ = paper_ssfa_object.run(X, J, mu=5)

    # Original
    U, Lam, UT = np.linalg.svd(np.cov(X))
    Q = U @ np.diag(Lam ** -(1/2))
    Z = Q.T @ X
    Z_dot = Z[:, 1:] - Z[:, :-1]
    P, Omega, PT = np.linalg.svd(np.cov(Z_dot))
    P = np.flip(P, axis=1)
    W_orig = (Q @ P)[:, :J]

    """Order Features and take slowest subset"""
    Y = W.T @ X
    Y_dot = Y[:, 1:] - Y[:, :-1]
    speeds = np.diag(Y_dot @ Y_dot.T) / n
    order = np.argsort(speeds)
    Omega_inv = np.diag(speeds[order] ** -1)
    W = W[:, order]

    Y_old = W_old.T @ X
    Y_dot_old = Y_old[:, 1:] - Y_old[:, :-1]
    speeds_old = np.diag(Y_dot_old @ Y_dot_old.T) / n
    order_old = np.argsort(speeds_old)
    Omega_inv_old = np.diag(speeds_old[order_old] ** -1)
    W_old = W_old[:, order_old]

    """Sparsity Calculation"""
    W_sparse = np.abs(W) / np.abs(W).sum(axis=0, keepdims=1)
    W_old_sparse = np.abs(W_old) / np.abs(W_old).sum(axis=0, keepdims=1)
    W_orig_sparse = np.abs(W_orig) / np.abs(W_orig).sum(axis=0, keepdims=1)

    # for i in range(W_sparse.shape[0]):
    #     for j in range(W_sparse.shape[1]):
    #         W_sparse[i, j] = W[i, j] / np.linalg.norm(W[:, j])
    #         W_old_sparse[i, j] = W_old[i, j] / np.linalg.norm(W_old[:, j])
    #         W_orig_sparse[i, j] = W_orig[i, j] / np.linalg.norm(W_orig[:, j])

    """Print Slowest Feature"""
    index_labels = [
        "XMEAS(01) A Feed  (stream 1) kscmh",
        "XMEAS(02) D Feed  (stream 2) kg/hr",
        "XMEAS(03) E Feed  (stream 3) kg/hr",
        "XMEAS(04) A and C Feed  (stream 4) kscmh",
        "XMEAS(05) Recycle Flow  (stream 8) kscmh",
        "XMEAS(06) Reactor Feed Rate  (stream 6) kscmh",
        "XMEAS(07) Reactor Pressure kPa gauge",
        "XMEAS(08) Reactor Level %",
        "XMEAS(09) Reactor Temperature Deg C",
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
        "XMV(01) D Feed Flow (stream 2)",
        "XMV(02) E Feed Flow (stream 3)",
        "XMV(03) A Feed Flow (stream 1)",
        "XMV(04) A and C Feed Flow (stream 4)",
        "XMV(05) Compressor Recycle Valve",
        "XMV(06) Purge Valve (stream 9)",
        "XMV(07) Separator Pot Liquid Flow (stream 10)",
        "XMV(08) Stripper Liquid Product Flow (stream 11)",
        "XMV(09) Stripper Steam Valve",
        "XMV(10) Reactor Cooling Water Flow",
        "XMV(11) Condenser Cooling Water Flow",
    ]
    lagged_labels = []
    for i in range(1, lagged_samples + 1):
        for lbl in index_labels:
            lagged_labels.append(f"{lbl} (t - {i})")
    index_labels += lagged_labels

    slowest = np.argsort(-1 * np.abs(W_sparse[:, 0]))
    slowest_old = np.argsort(-1 * np.abs(W_old_sparse[:, 0]))
    slowest_orig = np.argsort(-1 * np.abs(W_orig_sparse[:, 0]))
    print("Manifold Sparse SFA")
    for i in range(10):
        print(index_labels[slowest[i]],
              abs(W_sparse[:, 0][slowest[i]]))
    print("Sparse SFA")
    for i in range(10):
        print(index_labels[slowest_old[i]],
              abs(W_old_sparse[:, 0][slowest_old[i]]))
    print("SFA")
    for i in range(10):
        print(index_labels[slowest_orig[i]],
              abs(W_orig_sparse[:, 0][slowest_orig[i]]))

    """Sparsity Plot"""
    for threshold in thresholds:
        _f_sparse, axim = plt.subplots(nrows=1, ncols=3, sharey=True)
        _f_sparse.set_size_inches(21, 9)

        W_im = np.abs(np.copy(W_sparse))
        W_im_old = np.abs(np.copy(W_old_sparse))
        W_im_orig = np.abs(np.copy(W_orig_sparse))

        W_im[W_im <= threshold] = np.NaN
        W_im_old[W_im_old <= threshold] = np.NaN
        W_im_orig[W_im_orig <= threshold] = np.NaN

        s_pcnt = np.count_nonzero(np.isnan(W_im)) / np.size(W_im)
        s_pcnt_old = np.count_nonzero(np.isnan(W_im_old)) / np.size(W_im_old)
        s_pcnt_orig = np.count_nonzero(np.isnan(W_im_orig))/np.size(W_im_orig)
        axim[0].set_title(f"SFA ({s_pcnt_orig:.3f})")
        axim[1].set_title(f"Sparse SFA ({s_pcnt_old:.3f})")
        axim[2].set_title(f"Manifold Sparse SFA ({s_pcnt:.3f})")

        axim[0].imshow(np.abs(W_im_orig))
        axim[1].imshow(np.abs(W_im_old))
        im = axim[2].imshow(np.abs(W_im))

        axim[0].set_ylabel("Input Signal")
        axim[0].set_xlabel("Feature")
        axim[1].set_xlabel("Feature")
        axim[2].set_xlabel("Feature")

        _f_sparse.colorbar(im, ax=axim, aspect=60)

        if threshold == 0:
            magnitude = 0
            _f_sparse.text(0.2, 0.95, f'J = {J} | Threshold = 0', fontsize=24)
        else:
            magnitude = int(np.abs(np.floor(np.log10(threshold))))
            _f_sparse.text(0.2, 0.95, f'J = {J} | Threshold = 1e-{magnitude}',
                           fontsize=24)
        plt.savefig(f"plots/Sparsity_comparison_{magnitude}.png", dpi=350)
        plt.close(fig=_f_sparse)
        _f_sparse = None
