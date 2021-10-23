import numpy as np

from . import retraction
from . import proximal


class SSFA:
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

    def construct_finite_difference_matrix(self, n: int) -> np.ndarray:
        D = np.vstack(
                (
                    np.zeros((1, n - 1)),
                    np.eye(n - 1, n - 1)
                )
            ) - np.eye(n, n - 1)
        return(D)

    def cost_function(self, y: np.ndarray) -> np.ndarray:
        B = self.derivative_covariance
        f_y = np.trace(y.T @ B @ y)
        return(f_y)

    def cost_gradient(self, y: np.ndarray) -> np.ndarray:
        B = self.derivative_covariance
        grad_f_y = B @ y + y.T @ B
        return(grad_f_y)

    def cost_upper_bound(self, x: np.ndarray, y: np.ndarray,
                         eps: float) -> float:
        f_y = self.cost_function(y)
        grad_f_y = self.cost_gradient(y)
        x_minus_y = x - y
        l2_norm = np.linalg.norm(x_minus_y) ** 2

        f_hat = float(f_y + np.trace(grad_f_y.T @ x_minus_y)
                      + (0.5 * eps ** -1) * l2_norm)
        return(f_hat)

    def line_search(self, eps: float, y: np.ndarray, beta: float = 0.5,
                    max_iter: int = 50) -> tuple:
        for _ in range(max_iter):
            z = self.proximal_operator(y - eps * self.cost_gradient(y), eps)
            if self.cost_function(z) <= self.cost_upper_bound(z, y, eps):
                break
            eps *= beta
        return(eps, z)

    def run(self, X: np.ndarray, J: int, W: np.ndarray = None,
            max_iter: int = 500, err_tol: float = 1e-6):
        m, n = X.shape
        D = self.construct_finite_difference_matrix(n)

        A = np.cov(X)
        B = np.cov(X @ D)
        eps = 1 / (2 * np.linalg.norm(B))
        self.covariance = A
        self.derivative_covariance = B

        if W is None:
            W = np.eye(m, J)  # Initial guess

        converged = False
        tangent_norm = 1e-12
        sparsity_values = []
        relative_errors = []
        cost_values = []
        for k in range(max_iter):
            tangent = 2 * B @ W  # Derivative of cost function

            prev_norm = tangent_norm
            tangent_norm = np.linalg.norm(tangent)

            direction = -1 * tangent / tangent_norm
            alpha = self.step_size(k)
            W = self.retraction(W, A, alpha * direction)
            W = self.proximal_operator(W, eps)

            rel_error = abs(tangent_norm - prev_norm) / abs(prev_norm)
            sparsity_values.append(np.count_nonzero(W == 0) / np.size(W))
            relative_errors.append(rel_error)
            cost_values.append(self.cost_function(W))
            if rel_error < err_tol:
                print(f"Converged in {k} iterations with "
                      f"relative error of {rel_error}")
                converged = True
                break
        if not converged:
            print(f"Reached max iterations ({max_iter}) without converging, "
                  f"with final relative error of {rel_error}")
        return(W, cost_values, sparsity_values, relative_errors)
