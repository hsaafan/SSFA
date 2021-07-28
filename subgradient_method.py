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


X, T0, T4, T5, T10 = tep_import.import_tep_sets()

X = X - np.mean(X, axis=1).reshape((-1, 1))

L, lam, LT = np.linalg.svd(np.cov(X))
inv_lam = np.diag(lam**(-1/2))

Q = L @ inv_lam

X_dot = X[:, 1:] - X[:, :-1]
Z_dot = Q.T @ X_dot

P, Omega, PT = np.linalg.svd(np.cov(Z_dot))
W = (Q @ P).T

m = X.shape[0]
# w_j = W[-1, :]
# How this is initialized greatly affects results
w_j = np.random.rand(m, 1)
# w_j = np.ones((m, 1))

# Hyperparameters
# min (1/2)||w_j X_dot||^2_2 + lam_1 ||w_j||_1 + (lam_2 / 2) ||w_j||^2_2
# gamma is the gradient step size in proximal gradient descent
# gamma = 2e-5, lam_1 = 10, lam_2 = 1
gamma = 2e-5
lam_1 = 10
lam_2 = 1

A = X_dot @ X_dot.T
B = (1 - (lam_2 / lam_1) * gamma) * np.eye(m) - (gamma / lam_1) * A

# Convergence parameters
convergence_tolerance = 1e-4
max_iter = 5000
iter = 0
error = 1

while error > convergence_tolerance:
    iter += 1
    w_j_prev = np.copy(w_j)
    w_j = soft_thresholding(B @ w_j, gamma)
    if iter >= max_iter:
        print("Reached max iterations")
        break
    error = np.linalg.norm(w_j_prev - w_j) / np.linalg.norm(w_j_prev)

w_j = w_j / np.linalg.norm(w_j)

y_j = w_j.T @ X
print(w_j.reshape((-1,)))
w_j = w_j.flat
ssfa_speed = np.linalg.norm(y_j) / y_j.size
print(f"Zero weights: {m - np.count_nonzero(w_j)}")
print(f"Speed: {ssfa_speed}")

_f1, ax1 = plt.subplots()
ax1.plot((W @ X)[-1, :], label=f'Slowest from SFA - Speed = {Omega[-1]}')
ax1.plot(y_j.flat, label=f'Sparse Output - Speed = {ssfa_speed}')
ax1.legend()
ax1.set_title("Signal Outputs")
ax1.set_xlabel("Sample")
ax1.set_ylabel("$y_1(t)$")

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
