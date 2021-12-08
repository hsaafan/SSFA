import numpy as np


def construct_finite_difference_matrix(n: int) -> np.ndarray:
    """Backward difference matrix to estimate derivative"""
    D = np.vstack(
            (
                np.zeros((1, n - 1)),
                np.eye(n - 1, n - 1)
            )
        ) - np.eye(n, n - 1)
    return(D)


class SFA:
    def run(self, X: np.ndarray, J: int):
        m, n = X.shape
        D = construct_finite_difference_matrix(n)

        self.covariance = np.cov(X)
        self.derivative_covariance = np.cov(X @ D)

        U, Lam, UT = np.linalg.svd(self.covariance)
        Q = U @ np.diag(Lam ** -(1/2))
        Z = Q.T @ X
        Z_dot = Z[:, 1:] - Z[:, :-1]
        P, Omega, PT = np.linalg.svd(np.cov(Z_dot))
        W = (Q @ P)[:, :J]
        Omega_inv = np.diag(Omega[:J] ** -1)

        return(W, Omega_inv)
