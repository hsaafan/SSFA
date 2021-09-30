import numpy as np


def hard_threshold(array: np.ndarray, eps: float) -> np.ndarray:
    # Hard thresholding
    X = np.copy(array)
    X[np.abs(X) < eps] = 0
    return(X)


def soft_threshold(array: np.ndarray, eps: float) -> np.ndarray:
    # Soft thresholding used as proximal operator of l1 norm
    X = np.copy(array)
    X[np.abs(X) < eps] = 0
    X[X > eps] -= eps
    X[X < -eps] += eps
    return(X)


def shrinkage_operator(array: np.ndarray, eps: float) -> np.ndarray:
    # Shrinkage operator used as proximal operator of l2 norm
    return((1 / (1 + eps)) * array)


def elastic_net_operator(array: np.ndarray, eps: float,
                         gamma: float = 1) -> np.ndarray:
    # Elastic net proximal operator where gamma is an additional parameter
    # that is twice the coefficient of the l2 norm in the cost function
    return(shrinkage_operator(soft_threshold(array, eps), eps * gamma))
