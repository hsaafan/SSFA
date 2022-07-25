import src.sfamanopt.mssfa as mssfa
import numpy as np
import matplotlib.pyplot as plt

from scipy import io

if __name__ == "__main__":
    lagged_samples = 3
    features_plotted = 20
    # Algorithm names for labels
    """Import Data"""

    X0 = io.loadmat('proc1_45.mat')['proc1data45'].T

    X = np.copy(X0)
    for i in range(1, lagged_samples + 1):
        rolled = np.roll(X0, i, axis=1)
        X = np.append(X, rolled, axis=0)
    X = np.delete(X, range(lagged_samples), axis=1)

    m = X.shape[0]
    n = X.shape[1]
    X_mean = np.mean(X, axis=1).reshape((-1, 1))
    X = X - X_mean
    # X_std = np.std(X, axis=1).reshape((-1, 1))
    # X_norm = X / X_std

    """Train Models"""
    mssfa_object = mssfa.MSSFA("chol", "l1")
    W_mssfa, _, _, _, _ = mssfa_object.run(X, m)

    Y = W_mssfa.T @ X

    """Plot the features"""
    _f, axs2d = plt.subplots(nrows=features_plotted, ncols=1, sharex='col')
    _f.set_size_inches(15, 21)
    fontsize = 20

    for i in range(features_plotted):
        axs2d[i].set_title(f'Feature {i + 1}')
        axs2d[i].plot(Y[i, :])

    _f.set_tight_layout(True)
    plt.savefig(f"plots/proc_data.png", dpi=350)
    plt.close(fig=_f)
    _f = None
