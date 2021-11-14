import src.sfamanopt.ssfa as ssfa
import src.sfamanopt.paper_ssfa as oldssfa
import src.sfamanopt.fault_diagnosis as fd

import numpy as np
import matplotlib.pyplot as plt

import tepimport

if __name__ == "__main__":
    alpha = 0.01
    J = 33  # How many to calculate
    lagged_samples = 2
    thresholds = [0, 0.01, 0.001, 1e-6, 1e-12]
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
    W_sparse = np.zeros_like(W)
    W_old_sparse = np.zeros_like(W_old)
    W_orig_sparse = np.zeros_like(W_orig)

    for i in range(W_sparse.shape[0]):
        for j in range(W_sparse.shape[1]):
            W_sparse[i, j] = W[i, j] / np.linalg.norm(W[:, j])
            W_old_sparse[i, j] = W_old[i, j] / np.linalg.norm(W_old[:, j])
            W_orig_sparse[i, j] = W_orig[i, j] / np.linalg.norm(W_orig[:, j])

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
