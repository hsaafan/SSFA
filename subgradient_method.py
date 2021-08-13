import time
import numpy as np
import matplotlib.pyplot as plt
import tep_import
from math import log


def soft_thresholding(v: np.ndarray, t: float) -> np.ndarray:
    v_star = np.zeros_like(v)
    for i in range(v.size):
        if v[i] > t:
            v_star[i] = v[i] - t
        elif v[i] < -t:
            v_star[i] = v[i] + t

    return(v_star)


def accelerated_pgd(grad_g, x, gamma=2e-5, max_iter=1000, conv_tol=1e-6):

    iter = 0
    error = 1
    while error > conv_tol:
        wk = iter / (iter + 3)  # Acceleration parameter

        x_prev = np.copy(x)
        y = x + wk * (x - x_prev)
        step = y - gamma * grad_g(y)

        x = soft_thresholding(step.reshape((-1)), gamma).reshape(y.shape)

        iter += 1
        if iter >= max_iter:
            print("Reached max iterations")
            break
        # error = np.linalg.norm(x - x_prev) / np.linalg.norm(x_prev)
        error = np.linalg.norm(step)
    return(x)


X, T0, T4, T5, T10 = tep_import.import_tep_sets()
m = X.shape[0]
n = X.shape[1]

X = X - np.mean(X, axis=1).reshape((-1, 1))

# Convergence parameters
conv_tol = 0.2
max_iter = 10000


all_gradients = []
all_f_values = []

gamma = 2e-5
beta = 0.5

W = None
j = 2

X_orig = np.copy(X)
start = time.time()
for i in range(j):
    # w_j = np.random.rand(1, m)
    w_j = np.ones((1, m))
    X_dot = X[:, 1:] - X[:, :-1]
    A = (X_dot @ X_dot.T) / n
    B = (X @ X.T) / n
    mu = 1

    while True:
        gradients = []
        f_values = []

        def grad_g(x):
            y1 = x @ A
            xB = x @ B
            c1 = xB @ x.T - 1
            y2 = -1 * (mu / c1) * xB
            gradients.append(np.linalg.norm(y1 + y2))
            f_values.append(0.5 * float(y1 @ x.T - mu * log(xB @ x.T - 1)))
            return(y1 + y2)

        w_j = accelerated_pgd(grad_g, w_j, gamma, max_iter, conv_tol)
        all_gradients.append(gradients)
        all_f_values.append(f_values)
        mu *= beta
        if mu <= 1e-16:
            break

    scalar_projections = (w_j @ X) / (w_j @ w_j.T)
    vector_projections = np.hstack([w_j.T] * n)
    X = X - vector_projections @ np.diag(scalar_projections.flat)
total_time = time.time() - start
X = X_orig

"""-------------------- Results and Plotting --------------------"""
# Sparse results
print(f"Convereged in {total_time} seconds")
y_j = w_j @ X
print(w_j.reshape((-1,)))
w_j = w_j.flat
ssfa_speed = np.linalg.norm(y_j) / y_j.size
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
for i, data in enumerate(all_f_values):
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
