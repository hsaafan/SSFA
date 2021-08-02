import numpy as np
import numpy.linalg as npla
import tep_import
import matplotlib.pyplot as plt

import warnings
import math


def soft_threshold_vector(vector, eps):
    # Soft thresholding used as proximal operator of l1 norm
    x = np.copy(vector)
    for i in range(x.shape[0]):
        if x[i] >= eps:
            x[i] = x[i] - eps
        elif x[i] <= -eps:
            x[i] = x[i] + eps
        else:
            x[i] = 0
    return(x)


def fista(A, b, mu, maxit):
    # FISTA algorithm taken from:
    # https://gist.github.com/agramfort/ac52a57dc6551138e89b
    # Changed to use my soft threshold function
    x = np.zeros(A.shape[1])
    t = 1
    z = x.copy()
    L = npla.norm(A) ** 2
    for _ in range(maxit):
        xold = x.copy()
        z = z + A.T.dot(b - A.dot(z)) / L
        x = soft_threshold_vector(z, mu / L)
        t0 = t
        t = (1. + np.sqrt(1. + 4. * t ** 2)) / 2.
        z = x + ((t0 - 1.) / t) * (x - xold)

    return(z)


def optimize_P(X, Y, P, R, L, inv_lam, mu):
    """ My implementation
    # Cache matrices to save time
    to_z = L @ inv_lam
    to_p = npla.inv(to_z)
    A = X.T @ X

    # Copied how the other fista implementation found eps
    ls = npla.norm(X.T)
    eps = 1 / ls

    for col in range(P.shape[1]):
        Y_star = Y @ R[:, col]

        # l2-norm
        def fun(z):
            return(npla.norm(Y_star - X @ z) ** 2)

        # Proximal operator of l1 norm
        def prox_fun(z):
            proximal = soft_threshold_vector(z, eps)
            return(proximal)

        # # Define new gradient based on column of matrix
        def grad_l2(z):
            return(2 * (A @ z - X.T @ Y_star))

        # Change of variable to z
        z = to_z @ P[:, col]
        Optimize z using accelerated proximal gradient method
        z_opt = prox_grad(z, fun, grad_l2, prox_fun, lam)
        """

    # Using fista from github
    to_z = L @ inv_lam
    to_p = npla.inv(to_z)
    for col in range(P.shape[1]):
        Y_star = Y @ R[:, col]
        z_opt = fista(X, Y_star, mu, 100)
        # z_opt = fista_backtracking(Y_star, X, mu)
        # Change of variable back to p
        P[:, col] = to_p @ z_opt
    return(P)


def optimize_R(Y, P):
    # Formula from paper
    U, D, VT = npla.svd(Y.T @ Y @ P)
    R = U @ VT
    return(R)


def fista_backtracking(Y_star, X, mu, L=10, eta=1.5,
                       max_iters=500, tol=1e-3):
    z = np.zeros(X.shape[1])
    y = np.copy(z)
    t_next = 1

    def prox(z, L, mu):
        x = z + (2/L) * X.T @ (Y_star - X @ z)
        x = soft_threshold_vector(x, mu/L)
        return(x)

    def fun(z):
        return(npla.norm(Y_star - X @ z)**2 + mu * npla.norm(z, ord=1))

    def quad_min(x, y, L):
        fy = npla.norm(Y_star - X @ y)**2
        diff = x - y
        gradfy = 2 * (X.T @ X @ z - X.T @ Y_star)
        inprod = diff.T @ gradfy
        l2 = (L/2) * npla.norm(diff) ** 2
        gx = npla.norm(x, ord=1)
        return(fy + inprod + l2 + gx)

    z_prev = np.zeros_like(y)
    for i in range(max_iters):
        # Backtracking to find Lipschitz constant
        for eta_pwr in range(max_iters):
            eta_temp = eta ** eta_pwr
            prox_y = prox(y, eta_temp * L, mu)

            if fun(prox_y) <= quad_min(prox_y, y, eta_temp*L):
                L = eta_temp * L
                break

        z_prev = np.copy(z)
        z = prox(y, L, mu)

        t_prev = t_next
        t_next = (1 + math.sqrt(1 + 4 * t_prev ** 2))/2

        y = z + (t_prev - 1)/t_next * (z - z_prev)
        if i != 0 and npla.norm(z - z_prev) <= tol:
            break
    return(z, L)


def loss(Y, P, R, L, inv_lam):
    l1_loss = 0
    for col in range(P.shape[1]):
        l1_loss += npla.norm(L @ inv_lam @ P[:, col], ord=1)

    l2_loss = 0
    for row in range(Y.shape[0]):
        l2_loss += npla.norm(Y[row, :] - R @ P.T @ Y[row, :]) ** 2

    return(l1_loss, l2_loss)


def ssfa(mu=5, tol=1e-4, max_iters=500):
    # Import data sets
    X, T0, T4, T5, T10 = tep_import.import_tep_sets()

    # Add 1 dynamic copy to X (x(t-1))
    x_temp = X.T[:-1, :]
    X = np.append(x_temp, X.T[1:, :], axis=1)

    # Center data
    xmeans = X.mean(axis=0).reshape((1, -1))
    X = X - xmeans

    # Whiten Data
    Xdot = np.diff(X, axis=0)
    cov_Xdot = np.cov(Xdot, rowvar=False)
    L, lam, LT = npla.svd(cov_Xdot)
    inv_lam = np.diag(lam**(-1/2))
    y = inv_lam @ LT @ X.T  # columns are samples
    Y = y.T  # rows are samples

    # Initialize R and P matrices, might be a better way to do this
    R = np.eye(y.shape[0])
    P = np.eye(y.shape[0])

    # Loop variables
    prev_loss = 2
    curr_loss = 1
    k = 0
    l1_losses = []
    l2_losses = []
    # Perform sparse iterations
    while abs(prev_loss - curr_loss) / prev_loss > tol:
        k += 1
        prev_loss = curr_loss

        # Alternating optimization
        P = optimize_P(X, Y, P, R, L, inv_lam, mu)
        R = optimize_R(Y, P)

        # Calculate losses
        l1_loss, l2_loss = loss(Y, P, R, L, inv_lam)
        curr_loss = mu * l1_loss + l2_loss

        l1_losses.append(mu * l1_loss)
        l2_losses.append(l2_loss)

        if k > max_iters:
            plt.title("Cannot Converge, Showing Losses")
            plt.plot(l1_losses, label="$\mu l_1$")
            plt.plot(l2_losses, label="$l_2$")
            plt.legend()
            plt.show()
            raise RuntimeError("Cannot converge")
    print(f"Converged in {k} iterations")

    # Normalize P matrix to get Q
    for col in range(P.shape[1]):
        P[:, col] /= npla.norm(P[:, col])

    # Convert Q (normalized P) to W
    W = L @ inv_lam @ P

    # Calculate features, flip it since output is in reverse order
    S = np.flip(W.T @ X.T, axis=0)

    # Test statistics
    K = 30  # from paper
    tests = []
    for name, test in [("IDV(0)", T0), ("IDV(4)", T4),
                       ("IDV(5)", T5), ("IDV(10)", T10)]:
        # Prep test sets
        temp = test[:, :-1]
        test = np.append(temp, test[:, 1:], axis=0)
        test = test.T - xmeans

        # Calculate features, flip like above
        S_test = np.flip(W.T @ test.T, axis=0)
        S_test = S_test[:K, :]

        # Calculate t squared
        tsqr = []
        for i in range(S_test.shape[1]):
            tsqr.append(S_test[:, i].T @ S_test[:, i])
        tests.append((name, tsqr))
    return(S, W, l1_losses, l2_losses, tests)


if __name__ == "__main__":
    with warnings.catch_warnings():
        # Detect warnings as errors to prevent numpy NaN values
        warnings.simplefilter('error')
        S, W, l1_losses, l2_losses, tests = ssfa()

    # Plot number of non-zeros over slow features
    plt.figure("Models")
    plt.subplot(4, 1, 1)
    plt.title("Non-Zero Coefficients")
    plt.plot(np.sum(abs(W.T) > 1e-4, axis=1))

    # Plot coefficient values of first couple of features
    plt.subplot(4, 1, 2)
    plt.title("Sparse Coefficients")
    features = 3
    for i in range(features):
        plt.plot(W[:, i], label=f'{i}')
    plt.legend()

    # Plot the iteration losses over time
    plt.subplot(4, 1, 3)
    plt.title("Regression Losses")
    plt.plot(l2_losses)

    plt.subplot(4, 1, 4)
    plt.title("Sparsity Losses")
    plt.plot(l1_losses, label="$\mu l_1$")
    plt.legend()

    # Plot the first couple of features
    features = 6
    plt.figure("Features")
    for i in range(features):
        plt.subplot(features, 1, i + 1)
        plt.plot(S[i, :])

    # Plot the test statistics
    plt.figure("Tests")
    num_tests = len(tests)
    for i in range(num_tests):
        plt.subplot(num_tests, 1, i + 1)
        plt.title(tests[i][0])
        plt.plot(tests[i][1])
    plt.show()
