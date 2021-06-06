import numpy as np
import numpy.linalg as npla
import tep_import
import matplotlib.pyplot as plt
import warnings


def soft_threshold_vector(x: np.ndarray, eps: float):
    # Soft thresholding used as proximal operator of l1 norm
    for i in range(x.shape[0]):
        if x[i] >= eps:
            x[i] = x[i] - eps
        elif x[i] <= -eps:
            x[i] = x[i] + eps
        else:
            x[i] = 0
    return(x)


def optimize_P(X, Y, P, R, L, inv_lam):
    # FIXME arbitrary values
    lam = 0.001
    eps = 1e-6

    # Cache matrices
    A = X @ X.T
    to_z = L @ inv_lam
    to_p = npla.inv(inv_lam) @ npla.inv(L)

    # Proximal operator of l1 norm
    def prox_fun(z):
        proximal = soft_threshold_vector(z, eps)
        return(proximal)

    for col in range(P.shape[1]):
        Y_star = Y.T @ R[:, col]

        # Define new gradient based on column of matrix
        def grad_l2(z):
            gradient = z - 2 * lam * (A @ z - X @ Y_star)
            return(gradient)

        # Change of variable to z
        z = to_z @ P[:, col]
        # Optimize z using accelerated proximal gradient method
        z_opt = prox_grad(z, grad_l2, prox_fun)
        # Change of variable back to p
        P[:, col] = to_p @ z_opt
    return(P)


def optimize_R(Y, P):
    # Formula from paper
    U, D, VT = npla.svd(Y @ Y.T @ P)
    R = VT @ U
    return(R)


def prox_grad(x, grad_fun, prox_fun, rel_tol=1e-3, max_iters=500):
    # Iteration variables
    err = 2 * rel_tol
    k = 0
    diff = None

    # Need 1 past x value, get proximal of gradient once
    x_prev = x
    x = prox_fun(grad_fun(x))

    while err > rel_tol:
        # Iteration update
        k += 1
        if k > max_iters:
            raise RuntimeError(f"Reached {max_iters} iterations")
        # Tuning parameter update
        w = k / (k + 3)
        # Weighting
        y = x + w * (x - x_prev)
        # Proximal of gradient
        x = prox_fun(grad_fun(y))

        # Check convergence
        diff_old = diff
        diff = np.sum(abs(x - x_prev))
        if diff_old is None:
            continue
        err = abs(diff - diff_old) / diff_old
    return(x)


def loss(Y, P, R, L, inv_lam):
    l1_loss = 0
    for col in range(P.shape[1]):
        l1_loss += npla.norm(L @ inv_lam @ P[:, col], ord=1)

    l2_loss = 0
    for col in range(Y.shape[1]):
        l2_loss += npla.norm(Y[:, col] - R @ P.T @ Y[:, col]) ** 2

    return(l1_loss, l2_loss)


def ssfa(mu=10, tol=1e-3):
    # Import data sets
    X, T0, T4, T5, T10 = tep_import.import_tep_sets()
    # Center data
    xmeans = X.mean(axis=1).reshape((-1, 1))
    X -= xmeans
    # Whiten Data
    cov_X = np.cov(X)
    U, D, UT = npla.svd(cov_X)
    Q = U @ np.diag(D**-(1/2))
    X = Q.T @ X

    # Calculate original transformation
    Xdot = np.diff(X)
    cov_Xdot = np.cov(Xdot)
    L, Lambda, LT = npla.svd(cov_Xdot)
    order = Lambda.argsort()
    Lambda = Lambda[order]
    L = L[:, order]
    inv_lam = np.diag(Lambda**-(1/2))
    W_orig = L.T

    # Calculate original slow features
    Y_orig = L.T @ X

    R = np.eye(Y_orig.shape[0])
    P = np.eye(Y_orig.shape[0])

    prev_loss = 2
    curr_loss = 1
    k = 0
    # Perform sparse iterations
    while abs(prev_loss - curr_loss) / prev_loss > tol:
        k += 1
        prev_loss = curr_loss
        P = optimize_P(X, Y_orig, P, R, L, inv_lam)
        R = optimize_R(Y_orig, P)
        l1_loss, l2_loss = loss(Y_orig, P, R, L, inv_lam)
        curr_loss = mu * l1_loss + l2_loss
        if k > 1000:
            raise RuntimeError("Cannot converge")
    print(f"Converged in {k} iterations")

    # Normalize sparse transformation matrix
    W = np.zeros_like(P)
    for col in range(P.shape[1]):
        W[:, col] = P[:, col] / npla.norm(P[:, col])

    # Calculate sparse slow features
    Y = W.T @ X
    return(Y_orig, Y, W_orig, W.T, Lambda)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        Y_orig, Y, model, model_sparse, speeds_orig = ssfa()

    plt.figure("Models")
    plt.subplot(3, 1, 1)
    s_zeros = np.sum(abs(model_sparse) > 1e-6, axis=1)
    o_zeros = np.sum(abs(model) > 1e-6, axis=1)
    plt.plot(s_zeros, label="Sparse")
    plt.plot(o_zeros, label="Original")
    plt.legend()
    plt.title("Non-Zero Coefficients")

    plt.subplot(3, 1, 2)
    plt.title("Sparse Coefficients")
    plt.plot(model_sparse)

    plt.subplot(3, 1, 3)
    plt.title("Original Coefficients")
    plt.plot(model)

    # eta = np.around(Y.shape[1]/(2*np.pi) * np.sqrt(speeds_orig), 2)
    features = 6
    plt.figure("Features")
    for i in range(features):
        plt.subplot(features, 1, i + 1)
        plt.plot(Y[i, :], label="Sparse")
        plt.plot(Y_orig[i, :], label="Original")
        if i == 0:
            plt.legend()
    plt.show()
