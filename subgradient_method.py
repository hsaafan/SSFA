import time
import numpy as np
import matplotlib.pyplot as plt
import tep_import
# Good Seeds
np.random.seed(3534848)
# np.random.seed(282829)
# np.random.seed(35231238)
# np.random.seed(94219)
# np.random.seed(531784)

# Bad Seeds
# np.random.seed(5)
# np.random.seed(2652352)
# np.random.seed(2246)
# np.random.seed(98123)


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


def admm(prox_g, prox_h, gamma, m, max_iter=10000, conv_tol=1e-5):
    iter = 0
    error = 1
    z = np.random.rand(m)
    u = np.random.rand(m)
    while error > conv_tol:
        x = prox_g(z - u, gamma)
        z = prox_h(x + u, gamma)
        u = u + x - z

        iter += 1
        if iter >= max_iter:
            print("Reached max iterations")
            break
        error = np.linalg.norm(x) / np.linalg.norm(z)
    return(x, z)


X, T0, T4, T5, T10 = tep_import.import_tep_sets()
m = X.shape[0]
n = X.shape[1]

X = X - np.mean(X, axis=1).reshape((-1, 1))
# X = X / np.std(X, axis=1).reshape((-1, 1))

X_dot = X[:, 1:] - X[:, :-1]
A = (X_dot @ X_dot.T) / n
B = (X @ X.T) / n

# Hyperparameters
# min (1/2n)||w_j X_dot||^2_2 + lam_1 ||w_j||_1 + (lam_2 / 2) ||w_j||^2_2
# gamma is the gradient step size in proximal gradient descent

# Convergence parameters
convergence_tolerance = 0.2
max_iter = 5000

# Accelerated Proximal Gradient Descent
# Some good parameters: lam_1=0.1, lam_2=1, gamma=2e-5
lam_1 = 0.1
lam_2 = 1
lam_3 = 0.001
gamma = 2e-5
mu = 1
# How this is initialized greatly affects results
# w_j = np.random.rand(1, m)
w_j = np.ones((1, m))


# def grad_g(x):
#     y1 = x @ A
#     y2 = lam_2 * x
#     y = (y1 + y2) / lam_1
#     return(y)


# def grad_g(x):
#     xB = x @ B
#     y1 = x @ A
#     y2 = lam_2 * x
#     y3 = 2 * lam_3 * (xB @ x.T - 1) @ xB
#     y = (y1 + y2 + y3) / lam_1
#     return(y)


while True:
    def grad_g(x):
        y1 = x @ A
        y2 = x @ B
        c1 = x @ B @ x.T - 1
        y3 = -1 * (mu / c1) * y2
        return(y1 + y2 + y3)

    start = time.time()
    w_j = accelerated_pgd(grad_g, w_j, gamma, max_iter, convergence_tolerance)
    total_time = time.time() - start
    mu *= 0.5
    if mu <= 1e-16:
        break

# Alternating Direction Method of Multipliers
# Some good parameters: lam_1=0.1, lam_2=0.25, gamma=2e-5
# lam_1 = 0.1
# lam_2 = 0.25
# gamma = 2e-5
# B = gamma * A + (gamma * lam_2 + lam_1) * np.eye(m)
# B_inv = lam_1 * np.linalg.inv(B)


# def prox_g(v, gamma):
#     return(v @ B_inv)


# start = time.time()
# _, w_j = admm(prox_g, soft_thresholding, gamma, m)
# total_time = time.time() - start


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
ax1.plot((W @ X)[-1, :], label=f'Slowest from SFA - Speed = {Omega[-1]}')
ax1.plot(y_j.flat, label=f'Sparse Output - Speed = {ssfa_speed}')
ax1.legend()
ax1.set_title("Signal Outputs")
ax1.set_xlabel("Sample")
ax1.set_ylabel("$y_1(t)$")

# Plot weight contributions
_f2, ax2 = plt.subplots()
order = np.argsort(-1 * np.abs(w_j))
cum_percent = np.cumsum(np.abs(w_j)[order]) / np.sum(np.abs(w_j))
ordered_weights = np.abs(w_j)[order]
bar_labels = [str(x + 1) for x in order]

ax2.bar(bar_labels, ordered_weights)
ax3 = ax2.twinx()
ax3.plot(cum_percent, 'g')
ax2.set_title("Sparse Weights")

ax2.set_xlabel("$i$")
ax2.set_ylabel("$|(w_j)_i|$")
ax3.set_ylabel("Cumulative Weights")

plt.show()
