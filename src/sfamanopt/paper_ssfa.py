import numpy as np


class PaperSSFA:
    def __init__(self) -> None:
        pass

    def step_size(self, k: int, L: int = 3) -> float:
        alpha = k / (k + L)
        return(alpha)

    def construct_finite_difference_matrix(self, n: int) -> np.ndarray:
        D = np.vstack(
                (
                    np.zeros((1, n - 1)),
                    np.eye(n - 1, n - 1)
                )
            ) - np.eye(n, n - 1)
        return(D)

    def soft_thresholding(self, v: np.ndarray, t: float) -> np.ndarray:
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

    def fista(self, Yj: np.ndarray,
              X: np.ndarray,
              mu: float,
              z: np.ndarray,
              eps: float = 10,
              max_iter: int = 100,
              conv_tol: float = 1e-6) -> np.ndarray:

        z_prev = np.zeros_like(z)
        L = np.abs(2/mu) * np.linalg.norm(X.T @ X)
        eps = 1/L
        for k in range(max_iter):
            w_k = self.step_size(k, 3)
            y = z + w_k * (z - z_prev)
            z_prev = z
            step = y - eps * ((-2/mu) * X.T @ (Yj - X @ y))
            z = self.soft_thresholding(step, eps)
            if np.linalg.norm(z_prev - z) / np.linalg.norm(z_prev) < conv_tol:
                break

        return(z)

    def optimize_r(self, Y, P):
        # Formula from paper
        U, D, VT = np.linalg.svd(Y.T @ Y @ P)
        R = U @ VT
        return(R)

    def optimize_p(self, Y, P, R, p_to_z, z_to_p, X, mu):
        for j in range(R.shape[1]):
            Y_star = Y @ R[:, j].reshape((-1, 1))
            z_j = p_to_z @ P[:, j].reshape((-1, 1))
            z_j_optimal = self.fista(Y_star, X, mu, z_j).reshape((-1, ))
            P[:, j] = z_to_p @ z_j_optimal
        return(P)

    def overall_cost(self, P: np.ndarray,
                     R: np.ndarray,
                     p_to_z: np.ndarray,
                     Y: np.ndarray,
                     mu: float):
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
        L = L[:, :J]
        LT = LT[:J, :]
        Lambda = Lambda[:J]
        x_to_y = np.diag(Lambda ** -0.5) @ LT
        p_to_z = L @ np.diag(Lambda ** -0.5)
        z_to_p = np.diag(Lambda ** 0.5) @ LT

        X = X.T
        Y = np.zeros((n, J))
        for i in range(Y.shape[0]):
            Y[i, :] = x_to_y @ X[i, :]

        if P is None:
            P = np.eye(J, J)  # Initial guess
        if R is None:
            R = np.eye(J, J)

        converged = False
        sparsity_values = []
        relative_errors = []
        cost_values = []

        cost = self.overall_cost(P, R, p_to_z, Y, mu)
        for k in range(max_iter):
            P = self.optimize_p(Y, P, R, p_to_z, z_to_p, X, mu)
            R = self.optimize_r(Y, P)

            W = np.zeros((m, J))
            for j in range(P.shape[1]):
                q_j = P[:, j] / np.linalg.norm(P[:, j])
                W[:, j] = p_to_z @ q_j

            prev_cost = cost
            cost = self.overall_cost(R, P, p_to_z, Y, mu)

            rel_error = abs(cost - prev_cost) / abs(prev_cost)
            sparsity_values.append(np.count_nonzero(W == 0) / np.size(W))
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
