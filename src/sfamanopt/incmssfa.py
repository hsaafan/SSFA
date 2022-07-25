import numpy as np

from . import retraction
from . import proximal
from .mssfa import MSSFA


class IncMSSFA(MSSFA):
    def __init__(self, retraction_type: str = "chol",
                 sparsity_cost_type: str = "l1",
                 elastic_net_gamma: float = 1) -> None:
        super().__init__(retraction_type,
                         sparsity_cost_type,
                         elastic_net_gamma)
        self._m = 1
        self._J = 1
        self.clear(self._m, self._J)

    def _get_m(self) -> int:
        return(self._m)

    def _set_m(self, value: int) -> None:
        if value != self._m:
            self._m = value
            self.clear(value, self._J)
    m = property(fget=_get_m, fset=_set_m, doc="Number of input signals.")

    def _get_J(self) -> int:
        return(self._J)

    def _set_J(self, value: int) -> None:
        if value != self._J:
            self._J = value
            self.clear(self._m, value)
    J = property(fget=_get_J, fset=_set_J, doc="Number of slow features.")

    def clear(self, m: int, J: int) -> None:
        self.x_mean = np.zeros((m, 1))
        self.x_var = np.ones((m, 1))
        self.covariance = np.zeros((m, m))
        self.derivative_covariance = np.zeros((m, m))
        self.speeds = np.zeros((J, 1))

    def run(self, X: np.ndarray, J: int, W: np.ndarray = None,
            sparse_threshold: float = 1e-12,
            reorder_by_speed: bool = True,
            calculate_sparsity: bool = False,
            verbose: bool = False,
            L: int = 0) -> tuple:
        if verbose:
            print("Starting IncMSSFA...")
        m, n = X.shape
        self.clear(m, J)

        if W is None:
            W = np.eye(m, J)
        x = np.zeros((m, 1))
        y = np.zeros((J, 1))

        sparsity_values = []
        direction = np.ones_like(W)
        W_prev = np.zeros_like(W)
        for j in range(n):
            lr = max([1 / (j + 1 + L), 1e-4])
            fr = 1 - lr

            x_prev = x
            x_star = X[:, j].reshape((m, 1))
            self.x_mean = fr * self.x_mean + lr * x_star
            if j == 0:
                continue
            self.x_var = fr * self.x_var + lr * (x_star - self.x_mean) ** 2
            x = (x_star - self.x_mean) / (self.x_var ** 0.5)

            self.covariance = fr * self.covariance + lr * (x @ x.T)
            self.derivative_covariance = (fr * self.derivative_covariance
                                          + lr * (x - x_prev) @ (x - x_prev).T)
            eps = 1 / (2 * np.linalg.norm(self.derivative_covariance))

            # Derivative of cost function
            tangent = 2 * self.derivative_covariance @ W
            direction = -1 * tangent * eps
            # Proximal minimization followed by manifold optimization
            V = W + (j / (j + 3)) * (W - W_prev)
            W_prev = W
            A = self.covariance + 1e-6 * np.eye(m)  # Force positive definite
            W = retraction.chol_retraction(V, A, direction)
            W = proximal.soft_threshold(W, eps)

            y_prev = y
            y = W.T @ x
            y_dot = y - y_prev
            self.speeds = fr * self.speeds + lr * y_dot ** 2

            # Calculate sparsity
            s_vals = 0
            if calculate_sparsity:
                sparse_W = np.abs(W) / np.abs(W).sum(axis=0, keepdims=1)
                s_vals = np.count_nonzero(sparse_W <= sparse_threshold)
            # Add iteration stats to results
            sparsity_values.append(s_vals / np.size(W))

        speeds = self.speeds.reshape((-1))
        if reorder_by_speed:
            order = np.argsort(speeds)
            Omega_inv = np.diag(speeds[order] ** -1)
            W = W[:, order]
        else:
            Omega_inv = np.diag(self.speeds ** -1)

        self.W = W

        return(W, Omega_inv, sparsity_values)
