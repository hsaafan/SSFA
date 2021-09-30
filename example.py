import src.sfamanopt.ssfa as ssfa
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import tepimport


if __name__ == "__main__":
    """Import Data"""
    X, T0, T4, T5, T10 = tepimport.import_tep_sets(lagged_samples=2)
    m = X.shape[0]
    n = X.shape[1]
    X_mean = np.mean(X, axis=1).reshape((-1, 1))
    X = X - X_mean

    """Train Model"""
    ssfa_object = ssfa.SSFA("chol", "l1")
    W, sparsity_values, relative_errors = ssfa_object.run(X)

    n_test = T0.shape[1]
    Y0 = (W.T @ (T0 - X_mean))
    Y4 = (W.T @ (T4 - X_mean))
    Y5 = (W.T @ (T5 - X_mean))
    Y10 = (W.T @ (T10 - X_mean))

    T2 = np.zeros((4, n_test))
    for i in range(n_test):
        T2[0, i] = Y0[:, i].T @ Y0[:, i]
        T2[1, i] = Y4[:, i].T @ Y4[:, i]
        T2[2, i] = Y5[:, i].T @ Y5[:, i]
        T2[3, i] = Y10[:, i].T @ Y10[:, i]

    plt.subplot(4, 1, 1)
    plt.title("IDV(0)")
    plt.plot(T2[0, :], label="IDV(0)")

    plt.subplot(4, 1, 2)
    plt.title("IDV(4)")
    plt.plot(T2[1, :], label="IDV(4)")

    plt.subplot(4, 1, 3)
    plt.title("IDV(5)")
    plt.plot(T2[2, :], label="IDV(5)")

    plt.subplot(4, 1, 4)
    plt.title("IDV(10)")
    plt.plot(T2[3, :], label="IDV(10)")

    plt.figure()
    plt.plot([np.count_nonzero(W[:, i] == 0) for i in range(33)])
    plt.xlabel("Slow Features")
    plt.ylabel("Number of zero values")

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Sparse Values")
    plt.plot(sparsity_values)
    plt.subplot(1, 2, 2)
    plt.title("Relative Error")
    plt.plot(relative_errors[1:])
    plt.yscale('log')

    print(W)
    print(f'0 values of W: {np.count_nonzero(W==0)} / {np.size(W)}')
    plt.show()
