import numpy as np
from . import proximal
from .sfa import construct_finite_difference_matrix


class SSFA:
    def step_size(self, k: int, L: int = 3) -> float:
        """FISTA step size"""
        alpha = k / (k + L)
        return(alpha)

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

    def overall_cost_paper(self, P: np.ndarray,
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

    def overall_cost(self, y: np.ndarray) -> np.ndarray:
        B = self.derivative_covariance
        f_y = np.trace(y.T @ B @ y) + np.sum(np.abs(y))
        return(f_y)

    def convert_P_to_W(self, P: np.ndarray, p_to_z: np.ndarray):
        W = np.zeros_like(P)
        for j in range(P.shape[1]):
            q_j = P[:, j] / np.linalg.norm(P[:, j])
            W[:, j] = p_to_z @ q_j
        return(W)

    def run(self, X: np.ndarray,
            J: int,
            mu: float = 1,
            P: np.ndarray = None,
            R: np.ndarray = None,
            max_iter: int = 500,
            err_tol: float = 1e-6,
            sparse_pcnt: float = 0.01,
            reorder_by_speed: bool = True,
            verbose: bool = False):
        if verbose:
            print('Starting SSFA...')
        m, n = X.shape
        D = construct_finite_difference_matrix(n)

        self.covariance = np.cov(X)
        self.derivative_covariance = np.cov(X @ D)
        A = self.covariance
        B = self.derivative_covariance

        L, Lambda, LT = np.linalg.svd(self.derivative_covariance)

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

        W = self.convert_P_to_W(P, p_to_z)
        cost = self.overall_cost(W)
        for k in range(max_iter):
            """Alternatively optimize P and R"""
            P = self.optimize_p(Y, P, R, p_to_z, z_to_p, X, mu)
            R = self.optimize_r(Y, P)

            """Create W matrix to check sparsity criteria"""
            W_prev = W
            W = self.convert_P_to_W(P, p_to_z)

            """Collect information about algorithm performance"""
            prev_cost = cost
            cost = self.overall_cost(W)

            # rel_error = abs(cost - prev_cost) / abs(prev_cost)
            rel_error = (np.linalg.norm(W - W_prev) / np.linalg.norm(W_prev))

            """ Calculate sparsity
            Since $y_j = \sum_{i=0}^{m} W_{i, j} x_i$, we take sparse values as
            those that will contribute less than 1% to the norm of the
            transformation vector. Since Var(x_i) will also affect it's
            contribution, we also take that into consideration.
            """
            s_vals = 0
            for j in range(W.shape[1]):
                np.sum(np.abs(W[:, j] * np.diag(self.covariance)))
                threshold = (sparse_pcnt * np.linalg.norm(W[:, j])
                             / np.diag(self.covariance))
                s_vals += np.count_nonzero(W[:, j] <= threshold)
            sparsity_values.append(s_vals / np.size(W))
            relative_errors.append(rel_error)
            cost_values.append(cost)
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

        if reorder_by_speed:
            Y = W.T @ X.T
            Y_dot = Y[:, 1:] - Y[:, :-1]
            speeds = np.diag(Y_dot @ Y_dot.T) / n
            order = np.argsort(speeds)
            Omega_inv = np.diag(speeds[order] ** -1)
            W = W[:, order]

        return(W, Omega_inv, cost_values, sparsity_values, relative_errors)
