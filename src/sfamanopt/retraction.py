import numpy as np


def chol_retraction(X: np.ndarray,
                    A: np.ndarray,
                    zeta: np.ndarray) -> np.ndarray:
    """Perform a cholskey retraction onto the generalized Steifel manifold
    (X.T @ A @ X = I)"""
    eta = X + zeta
    Z = eta.T @ A @ eta
    R = np.linalg.cholesky(Z)
    X = eta @ np.linalg.inv(R.T)
    return(X)


def random_direction(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Project a random vector onto the tangent space to get a direction

    This function is purely for testing purposes to generate tangets to a
    generalized Steifel manifold (X.T @ A @ X = I)
    """
    Z = np.random.rand(*X.shape)
    XAZ = X.T @ A @ Z
    zeta = Z - X @ (0.5 * (XAZ + XAZ.T))
    return(zeta)


def orthogonality_metric(X: np.ndarray, A: np.ndarray) -> float:
    """Metric to determine how orthogonal a matrix is where 0 is a perfectly
    orthogonal matrix"""
    p = X.shape[1]
    Ident = np.eye(p)
    metric = np.linalg.norm(X.T @ A @ X - Ident) / np.linalg.norm(Ident)
    return(metric)


if __name__ == "__main__":
    n = 100
    p = 100
    A_0 = np.random.randn(n, p)
    A = A_0 @ A_0.T
    X = np.random.randn(n, p)
    for k in range(5):
        zeta = random_direction(X, A)
        X = chol_retraction(X, A, zeta)
        print(orthogonality_metric(X, A))
