import time
from math import log
from typing import Callable, Tuple

import numpy as np
import matplotlib.pyplot as plt

import tep_import


def soft_thresholding(v: np.ndarray,
                      t: float) -> np.ndarray:
    """
    Soft Thresholding Function
    v: np.ndarray
        The vector to perform thresholding
    t: float
        The threshold value
    """
    v_star = np.zeros_like(v)
    for i in range(v.size):
        if v[i] > t:
            v_star[i] = v[i] - t
        elif v[i] < -t:
            v_star[i] = v[i] + t

    return(v_star)


def backtracking(g: Callable[[np.ndarray], np.ndarray],
                 grad_g: Callable[[np.ndarray], np.ndarray],
                 prox_f: Callable[[np.ndarray, float], np.ndarray],
                 y: np.ndarray,
                 gamma: float,
                 beta: float = 0.5) -> Tuple[np.ndarray, float]:
    """
    Backtracking algorithm to find proximal function parameter

    Parameters:
    -----------
    g: function(np.ndarray) -> np.ndarray
        The differentiable part of the cost function
    grad_g: function(np.ndarray) -> np.ndarray
        A function that takes a vector of variables x and returns the gradient
        of the differentiable part of the optimization function
    prox_f: function(np.ndarray, float) -> np.ndarray
        The proximal function of the non-differentiable part of the
        optimization function which takes a vector of variables x, a proximal
        variable gamma, and returns a vector of variables
    y: np.ndarray

    gamma: float

    beta: float

    Outputs:
    --------
    z: np.ndarray

    gamma: float

    """
    def g_hat(x1, x2):
        c1 = g(x2)
        c2 = grad_g(x2) @ (x1 - x2).T
        c3 = 1 / (2 * gamma) * (x1 - x2) @ (x1 - x2).T
        return(float(c1 + c2 + c3))

    while True:
        step = (y - gamma * grad_g(y)).reshape((-1))
        z = prox_f(step, gamma).reshape((1, -1))
        if g(z) <= g_hat(z, y):
            break
        gamma *= beta
    return(z, gamma)


def accelerated_pgd(g: Callable[[np.ndarray], np.ndarray],
                    grad_g: Callable[[np.ndarray], np.ndarray],
                    prox_f: Callable[[np.ndarray, float], np.ndarray],
                    x: np.ndarray,
                    gamma: float,
                    max_iter: int,
                    conv_tol: float) -> np.ndarray:
    """
    Accelerated Proximal Gradient Descent Algorithm

    Parameters:
    -----------
    g: function(np.ndarray) -> np.ndarray
        The differentiable part of the cost function
    grad_g: function(np.ndarray) -> np.ndarray
        A function that takes a vector of variables x and returns the gradient
        of the differentiable part of the optimization function
    prox_f: function(np.ndarray, float) -> np.ndarray
        The proximal function of the non-differentiable part of the
        optimization function which takes a vector of variables x, a proximal
        variable gamma, and returns a vector of variables
    x: np.ndarray
        The vector of variables of the optimization function
    gamma: float

    max_iter: int
        The maximum number of iterations to run if the convergence tolerance
        isn't reached
    conv_tol: float
        Once the error is below this tolerance, the algorithm stops iterating

    Outputs:
    --------
    x: np.ndarray
        The proximal vector of variables
    gradients: array

    costs: array

    """

    iter = 0
    error = 1
    gradients = []
    costs = []
    while error > conv_tol:
        # Acceleration using previous vector
        wk = iter / (iter + 3)  # Acceleration parameter
        x_prev = np.copy(x)
        y = x + wk * (x - x_prev)

        x, gamma = backtracking(g=g,
                                grad_g=grad_g,
                                prox_f=prox_f,
                                y=y,
                                gamma=gamma)
        costs.append(g(x))
        gradients.append(np.linalg.norm(grad_g(x)))
        # Convergence checks
        iter += 1
        if iter >= max_iter:
            print("Reached max iterations")
            break
        error = float(np.linalg.norm(x - x_prev) / np.linalg.norm(x_prev))

    return(x, gradients, costs)


def sparse_SFA(cov_X: np.ndarray,
               cov_X_dot: np.ndarray,
               j: int,
               beta: float = 0.5,
               gamma: float = 1e-4,
               max_iter: int = 10000,
               conv_tol: float = 1e-3):
    """
    Sparse Slow Feature Analysis

    Parameters:
    -----------
    cov_X: np.ndarray
        The covariance matrix of the data
    cov_X_dot: np.ndarray
        The covariance matrix of the derivative of the data
    j: int
        The number of slow features to extract
    beta: float
        The scaling of the barrier variable after each iteration of the
        interior point method
    gamma: float
        The thresholding parameter
    max_iter: int
        The maximum number of iterations to run of proximal gradient descent
    conv_tol: float
        The error value at which to stop iterating proximal gradient descent

    Outputs:
    --------
    w_j: np.ndarray

    all_gradients: array

    all_costs: array

    """
    m = cov_X.shape[0]
    prev_W = np.ones((m, 1))
    for i in range(j):
        w_j = np.ones((1, m))
        mu = 1
        all_gradients = []
        all_costs = []

        while mu > 1e-16:
            if i == 0:
                def g(x):
                    y1 = x @ cov_X_dot @ x.T
                    try:
                        y2 = -1 * mu * log(x @ cov_X @ x.T - 1)
                    except ValueError:
                        y2 = 1e16
                    return(0.5 * float(y1 + y2))

                def grad_g(x):
                    xB = x @ cov_X
                    c1 = xB @ x.T - 1

                    y1 = x @ cov_X_dot
                    y2 = -1 * (mu / c1) * xB
                    return(y1 + y2)
            else:
                def g(x):
                    xBW = x @ cov_X @ prev_W
                    y1 = x @ cov_X_dot @ x.T
                    try:
                        y2 = -1 * mu * log(x @ cov_X @ x.T - 1)
                    except ValueError:
                        y2 = 1e16
                    y3 = xBW @ xBW.T
                    return(0.5 * float(y1 + y2 + y3))

                def grad_g(x):
                    xB = x @ cov_X
                    BW = cov_X @ prev_W
                    xBW = x @ BW

                    c1 = xB @ x.T - 1
                    # c2 = xBW @ xBW.T

                    y1 = x @ cov_X_dot
                    y2 = -1 * (mu / c1) * xB
                    y3 = xBW @ BW.T

                    return(y1 + y2 + y3)

            w_j, gradient, cost = accelerated_pgd(g=g,
                                                  grad_g=grad_g,
                                                  prox_f=soft_thresholding,
                                                  x=w_j,
                                                  gamma=gamma,
                                                  max_iter=max_iter,
                                                  conv_tol=conv_tol)

            all_gradients.append(gradient)
            all_costs.append(cost)
            mu *= beta

        if i == 0:
            prev_W = np.copy(w_j.T)
        else:
            prev_W = np.hstack((prev_W, w_j.T))
    return(w_j, all_gradients, all_costs)


"""------------------------- Import Data Sets ------------------------------"""
X, T0, T4, T5, T10 = tep_import.import_tep_sets()
m = X.shape[0]
n = X.shape[1] - 1

"""------------------------- Calculate Covariances -------------------------"""
X_mean = np.mean(X, axis=1).reshape((-1, 1))
X = X - X_mean
X_std = np.std(X, axis=1).reshape((-1, 1))
X = X / X_std

X_dot = X[:, 1:] - X[:, :-1]
cov_X_dot = (X_dot @ X_dot.T) / n
cov_X = (X @ X.T) / n

"""------------------------- Set Parameters --------------------------------"""
conv_tol = 1e-3
max_iter = 10000
gamma = 1
beta = 0.9
j = 4

"""------------------------- Results and Plotting --------------------------"""
# Sparse results
start = time.time()
w_j, all_gradients, all_costs = sparse_SFA(cov_X=cov_X,
                                           cov_X_dot=cov_X_dot,
                                           j=j,
                                           beta=beta,
                                           gamma=gamma,
                                           max_iter=max_iter,
                                           conv_tol=conv_tol)
total_time = time.time() - start

print(f"Convereged in {total_time} seconds")
y_j = w_j @ X
y_j_dot = y_j[:, 1:] - y_j[:, :-1]
print(w_j.reshape((-1,)))
w_j = w_j.flat
ssfa_speed = float(y_j_dot @ y_j_dot.T) / y_j.size
print(f"Zero weights: {m - np.count_nonzero(w_j)}")
print(f"Speed: {ssfa_speed}")

# Normal SFA for Comparison
L, lam, LT = np.linalg.svd(np.cov(X))
inv_lam = np.diag(lam**(-1/2))
Q = L @ inv_lam
Z_dot = Q.T @ X_dot
P, Omega, PT = np.linalg.svd(np.cov(Z_dot))
W = (Q @ P).T

# Compare SFA to sparse feature
_f1, ax1 = plt.subplots()
ax1.plot((W @ X)[-j, :], label=f'Slowest from SFA - Speed = {Omega[-j]}')
ax1.plot(y_j.flat, label=f'Sparse Output - Speed = {ssfa_speed}')
ax1.legend()
ax1.set_title("Signal Outputs")
ax1.set_xlabel("Sample")
ax1.set_ylabel("$y_1(t)$")

# Plot weight contributions
_f2, ax21 = plt.subplots()
order = np.argsort(-1 * np.abs(w_j))
cum_percent = np.cumsum(np.abs(w_j)[order]) / np.sum(np.abs(w_j))
ordered_weights = np.abs(w_j)[order]
bar_labels = [str(x + 1) for x in order]

ax21.bar(bar_labels, ordered_weights)
ax22 = ax21.twinx()
ax22.plot(cum_percent, 'g')
ax21.set_title("Sparse Weights")

ax21.set_xlabel("$i$")
ax21.set_ylabel("$|(w_j)_i|$")
ax22.set_ylabel("Cumulative Weights")

# Gradients over iterations
_f3, ax3 = plt.subplots()
for i, data in enumerate(all_gradients):
    if len(data) <= 10:
        continue
    ax3.plot(data, label=f"$\mu$ iteration {i}")
ax3.set_yscale('log')
ax3.legend()
ax3.set_xlabel("PGD Iteration")
ax3.set_title("Gradient Over Iterations")

# f values over iterations
_f4, ax4 = plt.subplots()
for i, data in enumerate(all_costs):
    if len(data) <= 10:
        continue
    ax4.plot(data, label=f"$\mu$ iteration {i}")
ax4.set_yscale('log')
ax4.legend()
ax4.set_xlabel("PGD Iteration")
ax4.set_title("$f$ Values Over Iterations")

# Original SFA weights
_f5, ax51 = plt.subplots()

sfa_order = np.argsort(-1 * np.abs(W[-j]))
sfa_cum_percent = np.cumsum(np.abs(W[-j])[sfa_order]) / np.sum(np.abs(W[-j]))
sfa_ordered_weights = np.abs(W[-j])[sfa_order]
sfa_bar_labels = [str(x + 1) for x in sfa_order]

ax51.bar(sfa_bar_labels, sfa_ordered_weights)
ax52 = ax51.twinx()
ax52.plot(sfa_cum_percent, 'r')
ax51.set_yscale('log')
ax51.set_title("Original SFA Normalized Weights")

ax51.set_xlabel("$i$")
ax51.set_ylabel("$|(w_j)_i|$")
ax52.set_ylabel("Cumulative Weights")

plt.show()
