import math
import numpy as np


def AIC(m: int, N: int, s: int, L: int,
        y: np.ndarray, pred: np.ndarray) -> float:
    """Calculate the akaike information criterion

    Parameters
    ----------
    m: int
        Number of measured outputs
    N: int
        Number of samples
    s: int

    L: int
        Number of measured inputs
    y: np.ndarray
        Measured outputs
    pred: np.ndarray

    Returns
    -------
    objective: float
        The calculated AIC value
    """
    pred_err = y - pred
    sigma = (1 / (N - s)) * np.sum(pred_err @ pred_err.T)
    M_s = 2 * s * m + (m * (m + 1)) / 2 + s * L + m * L
    delta_s = N / (N - (M_s / m + (m + 1) / 2))
    objective = ((N - s) * (m * (1 + math.log(2 * math.pi)) + math.log(sigma))
                 + 2 * delta_s * M_s)
    return(objective)


def BIC(N: int, s: int,
        y: np.ndarray, pred: np.ndarray) -> float:
    """Calculate the Bayesian information criterion

    Parameters
    ----------
    N: int
        Number of samples
    s: int

    y: np.ndarray

    pred: np.ndarray

    Returns
    -------
    objective: float
        The calculated BIC value
    """
    pred_err = y - pred
    objective = ((N - s) * math.log(np.sum(np.linalg.norm(pred_err)))
                 + 2 * s * math.log(N - s) * math.log(math.log(N - s)))
    return(objective)


def optimize_lag(m: int, N: int, L: int,
                 y: np.ndarray, pred: np.ndarray,
                 s_max: int, s_min: int = 0) -> dict:
    if s_min > s_max:
        raise ValueError(f'Minimum lag of {s_min} is greater than '
                         f'maximum lag of {s_max}')
    lags = []
    BIC_values = []
    AIC_values = []
    for s in range(s_min, s_max):
        lags.append(s)
        BIC_values.append(BIC(N, s, y, pred))
        AIC_values.append(AIC(m, N, s, L, y, pred))
    BIC_min = np.argmin(BIC_values)
    AIC_min = np.argmin(AIC_values)
    opt_values = zip(lags, BIC_values, AIC_values)

    return((lags[BIC_min], BIC_values[BIC_min]),
           (lags[AIC_min], AIC_values[AIC_min]),
           opt_values)
