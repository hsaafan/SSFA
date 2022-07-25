import numpy as np

from . import retraction
from . import proximal
from .sfa import construct_finite_difference_matrix


class MSSFA:
    def __init__(self, retraction_type: str = "chol",
                 sparsity_cost_type: str = "l1",
                 elastic_net_gamma: float = 1) -> None:
        self.set_retraction_type(retraction_type)
        self.set_sparsity_cost(sparsity_cost_type, elastic_net_gamma)

    def set_retraction_type(self, retraction_type: str) -> None:
        if retraction_type.lower() == "chol":
            self.retraction = retraction.chol_retraction
        # Other types go here
        else:
            raise ValueError(f"Unknown retraction type: {retraction_type}")

    def set_sparsity_cost(self, cost_type: str, en_gamma: float = 1) -> None:
        if cost_type.lower() == "l1":
            self.proximal_operator = proximal.soft_threshold
        elif cost_type.lower() == "l2":
            self.proximal_operator = proximal.shrinkage_operator
        elif cost_type.lower() == "elastic net":
            def en_prox(array: np.ndarray, eps: float):
                return(proximal.elastic_net_operator(array, eps, en_gamma))
            self.proximal_operator = en_prox
        else:
            raise ValueError(f"Unknown sparsity cost type: {cost_type}")

    def step_size(self, k: int, L: int = 3) -> float:
        alpha = 1 / (k + L)
        return(alpha)

    def overall_cost(self, y: np.ndarray) -> float:
        """Objective function value"""
        B = self.derivative_covariance
        f_y = np.trace(y.T @ B @ y) + np.sum(np.abs(y))
        return(f_y)

    def run(self, X: np.ndarray, J: int, W: np.ndarray = None,
            max_iter: int = 500, err_tol: float = 1e-6,
            sparse_pcnt: float = 0.01,
            reorder_by_speed: bool = True,
            verbose: bool = False) -> tuple:
        if verbose:
            print("Starting MSSFA...")
        m, n = X.shape
        D = construct_finite_difference_matrix(n)

        A = np.cov(X)
        B = np.cov(X @ D)
        eps = 1 / (2 * np.linalg.norm(B))
        self.covariance = A
        self.derivative_covariance = B

        if W is None:
            # W = np.ones((m, J)) + np.eye(m, J)  # Initial guess
            W = np.eye(m, J)

        converged = False
        cost = self.overall_cost(W)
        sparsity_values = []
        relative_errors = []
        direction = np.ones_like(W)
        W_prev = np.zeros_like(W)
        for k in range(max_iter):
            # Get direction to move in
            tangent = 2 * B @ W  # Derivative of cost function
            direction = -1 * tangent * eps

            # Proximal minimization followed by manifold optimization
            alpha = self.step_size(k)

            V = W + (k / (k + 3)) * (W - W_prev)
            W_prev = W
            W = self.retraction(V, A, alpha * direction)
            W = self.proximal_operator(W, eps)
            rel_error = (np.linalg.norm(W - W_prev) / np.linalg.norm(W_prev))

            """ Calculate sparsity
            Since $y_j = \sum_{i=0}^{m} W_{i, j} x_i$, we take sparse values as
            those that will contribute less than 1% to the norm of the
            transformation vector. Since Var(x_i) will also affect it's
            contribution, we also take that into consideration.
            """
            s_vals = 0
            for j in range(W.shape[1]):
                threshold = sparse_pcnt * np.linalg.norm(W[:, j])
                s_vals += np.count_nonzero(W[:, j] <= threshold)

            # Add iteration stats to results
            sparsity_values.append(s_vals / np.size(W))
            relative_errors.append(rel_error)

            # Check convergence
            if rel_error < err_tol:
                if verbose:
                    print(f"Converged in {k} iterations with "
                          f"relative error of {rel_error}")
                converged = True
                break
        if not converged and verbose:
            print(f"Reached max iterations ({max_iter}) without converging, "
                  f"with final relative error of {rel_error}")
        if verbose:
            print(f'Slowest Feature: {np.min(np.diag(W.T @ B @ W))}')

        Y = W.T @ X
        Y_dot = Y[:, 1:] - Y[:, :-1]
        speeds = np.diag(Y_dot @ Y_dot.T) / n
        if reorder_by_speed:
            order = np.argsort(speeds)
            Omega_inv = np.diag(speeds[order] ** -1)
            W = W[:, order]
        else:
            Omega_inv = np.diag(speeds ** -1)

        return(W, Omega_inv, sparsity_values, relative_errors)
