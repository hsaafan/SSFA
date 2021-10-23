import numpy as np
from . import proximal


class PaperSSFA:
    def step_size(self, k: int, L: int = 3) -> float:
        """FISTA step size"""
        alpha = k / (k + L)
        return(alpha)

    def construct_finite_difference_matrix(self, n: int) -> np.ndarray:
        """Backward difference matrix to estimate derivative"""
        D = np.vstack(
                (
                    np.zeros((1, n - 1)),
                    np.eye(n - 1, n - 1)
                )
            ) - np.eye(n, n - 1)
        return(D)

    def fista(self, Yj: np.ndarray,
              X: np.ndarray,
              mu: float,
              z: np.ndarray,
              eps: float,
              max_iter: int = 100) -> np.ndarray:
        """FISTA algorithm"""
        XTY = X.T @ Yj  # Calculate beforehand to save time on iterations
        XTX = X.T @ X

        z_prev = np.zeros_like(z)
        for k in range(max_iter):
            w_k = self.step_size(k, 3)
            y = z + w_k * (z - z_prev)
            z_prev = z
            step = y - eps * ((-2/mu) * (XTY - XTX @ y))
            z = proximal.soft_threshold(step, eps)
        return(z)

    def optimize_r(self, Y, P):
        """Formula from paper in paragraph under Equation 13"""
        U, D, VT = np.linalg.svd(Y.T @ Y @ P, full_matrices=False)
        R = U @ VT
        return(R)

    def optimize_p(self, Y, P, R, p_to_z, z_to_p, X, mu):
        """Perform FISTA for each column vector of P"""
        L = np.abs(2/mu) * np.linalg.norm(X.T @ X)  # Lipschitz constant
        for j in range(R.shape[1]):
            Y_star = Y @ R[:, j].reshape((-1, 1))
            z_j = p_to_z @ P[:, j].reshape((-1, 1))
            z_j_optimal = self.fista(Y_star, X, mu, z_j, 1/L).reshape((-1, ))
            P[:, j] = z_to_p @ z_j_optimal
        return(P)

    def overall_cost(self, P: np.ndarray,
                     R: np.ndarray,
                     p_to_z: np.ndarray,
                     Y: np.ndarray,
                     mu: float):
        """Equation 9 from paper"""
        l2_norm = 0
        for i in range(Y.shape[0]):
            y_t = Y[i, :].reshape((-1, 1))
            l2_norm += np.linalg.norm(y_t - R @ P.T @ y_t) ** 2
        l1_norm = 0
        for j in range(P.shape[1]):
            p_j = P[:, j].reshape((-1, 1))
            l1_norm = np.sum(np.abs(p_to_z @ p_j))
        return(l2_norm + mu * l1_norm)

    def run(self, X: np.ndarray,
            J: int,
            mu: float = 1,
            P: np.ndarray = None,
            R: np.ndarray = None,
            max_iter: int = 500,
            err_tol: float = 1e-6):
        m, n = X.shape
        D = self.construct_finite_difference_matrix(n)

        self.covariance = np.cov(X)
        self.derivative_covariance = np.cov(X @ D)

        L, Lambda, LT = np.linalg.svd(self.derivative_covariance)
        """Take J largest singular vectors/values (Same as K from paper)"""
        # L = L[:, :J]
        # LT = LT[:J, :]
        # Lambda = Lambda[:J]

        """Transformations from paper"""
        x_to_y = np.diag(Lambda ** -0.5) @ LT
        p_to_z = L @ np.diag(Lambda ** -0.5)
        z_to_p = np.diag(Lambda ** 0.5) @ LT

        """Paper has rows as samples instead of columns as samples"""
        X = X.T
        Y = np.zeros((n, m))
        for i in range(Y.shape[0]):
            Y[i, :] = x_to_y @ X[i, :]

        """Initial guesses"""
        if P is None:
            P = np.eye(m, J)
        if R is None:
            R = np.eye(m, J)

        """Convergence information and algorithm performance"""
        converged = False
        sparsity_values = []
        relative_errors = []
        cost_values = []

        cost = self.overall_cost(P, R, p_to_z, Y, mu)
        for k in range(max_iter):
            """Alternatively optimize P and R"""
            P = self.optimize_p(Y, P, R, p_to_z, z_to_p, X, mu)
            R = self.optimize_r(Y, P)

            """Create W matrix to check sparsity criteria"""
            W = np.zeros((m, J))
            for j in range(P.shape[1]):
                q_j = P[:, j] / np.linalg.norm(P[:, j])
                W[:, j] = p_to_z @ q_j

            """Collect information about algorithm performance"""
            prev_cost = cost
            cost = self.overall_cost(R, P, p_to_z, Y, mu)

            rel_error = abs(cost - prev_cost) / abs(prev_cost)
            sparsity_values.append(np.count_nonzero(W <= 1e-6) / np.size(W))
            relative_errors.append(rel_error)
            cost_values.append(cost)
            if rel_error < err_tol:
                print(f"Converged in {k} iterations with "
                      f"relative error of {rel_error}")
                converged = True
                break
        if not converged:
            print(f"Reached max iterations ({max_iter}) without converging, "
                  f"with final relative error of {rel_error}")
        return(W, cost_values, sparsity_values, relative_errors)
