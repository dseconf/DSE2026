"""Conditional moments, skewness, and their influence functions.

Port of influencefunctions.R. All influence functions return an (n, 1) array,
except beta_infl, which returns (n, k). Estimator functions return scalars,
except beta, which returns (k, 1).
"""

import numpy as np

__all__ = [
    "mean_cond", "cov_cond", "beta", "skew", "skew_cond",
    "mean_infl", "beta_infl", "cov_infl", "mean_cond_infl",
    "mean_ratio_infl", "cov_cond_infl", "skew_infl", "skew_cond_infl",
]


def _vec(a):
    return np.asarray(a, dtype=float).reshape(-1)


def _col(a):
    return np.asarray(a, dtype=float).reshape(-1, 1)


def _mat(a):
    a = np.asarray(a, dtype=float)
    return a if a.ndim == 2 else a.reshape(-1, 1)


# Conditional mean and covariance ---------------------------------------------

def mean_cond(x, dum):
    """E(x | dum = 1), dum in {0, 1}."""
    x, dum = _vec(x), _vec(dum)
    return np.mean(x * dum) / np.mean(dum)


def cov_cond(x, y, dum):
    """cov(x, y | dum = 1), dum in {0, 1}."""
    x, y, dum = _vec(x), _vec(y), _vec(dum)
    mu_dum = np.mean(dum)
    return (np.mean(x * y * dum) / mu_dum
            - (np.mean(x * dum) / mu_dum) * (np.mean(y * dum) / mu_dum))


def beta(x, y):
    """Slope of the linear regression y = x @ beta + u."""
    X, yv = _mat(x), _col(y)
    return np.linalg.solve(X.T @ X, X.T @ yv)


def skew(x):
    """E[(x - E[x])^3] / E[(x - E[x])^2]^(3/2)."""
    x = _vec(x)
    d = x - np.mean(x)
    return np.mean(d ** 3) / np.mean(d ** 2) ** 1.5


def skew_cond(x, dum):
    """Conditional skewness, dum in {0, 1}."""
    x, dum = _vec(x), _vec(dum)
    mu_dum = np.mean(dum)
    mu_c_x3 = np.mean(x ** 3 * dum) / mu_dum
    mu_c_x2 = np.mean(x ** 2 * dum) / mu_dum
    mu_c_x = np.mean(x * dum) / mu_dum
    F_dum = mu_c_x3 - 3 * mu_c_x2 * mu_c_x + 2 * mu_c_x ** 3
    G_dum = mu_c_x2 - mu_c_x ** 2
    return F_dum / G_dum ** 1.5


# Influence functions ---------------------------------------------------------

def mean_infl(x):
    """Influence function for the mean."""
    x = _vec(x)
    return _col(x - np.mean(x))


def beta_infl(x, y):
    """Influence function for OLS coefficients in y = x @ beta + u."""
    X, yv = _mat(x), _vec(y)
    n = yv.size
    b = np.linalg.solve(X.T @ X, X.T @ yv)
    u_hat = yv - X @ b
    # infl = (X * u_hat) @ inv(X'X / n); X'X is symmetric, so solve on the
    # transposed system avoids forming the inverse explicitly.
    return np.linalg.solve((X.T @ X) / n, (X * u_hat[:, None]).T).T


def cov_infl(x, y):
    """Influence function for cov(x, y)."""
    x, y = _vec(x), _vec(y)
    mu_xy, mu_x, mu_y = np.mean(x * y), np.mean(x), np.mean(y)
    return _col(x * y - mu_xy - mu_y * (x - mu_x) - mu_x * (y - mu_y))


def mean_cond_infl(x, dum):
    """Influence function for E(x | dum = 1), dum in {0, 1}."""
    x, dum = _vec(x), _vec(dum)
    mu_dum = np.mean(dum)
    mu_x_dum = np.mean(x * dum)
    return _col((mu_dum * (x * dum - mu_x_dum)
                 - mu_x_dum * (dum - mu_dum)) / mu_dum ** 2)


def mean_ratio_infl(x, y):
    """Influence function for E(x) / E(y)."""
    x, y = _vec(x), _vec(y)
    mu_x, mu_y = np.mean(x), np.mean(y)
    return _col((mu_y * (x - mu_x) - mu_x * (y - mu_y)) / mu_y ** 2)


def cov_cond_infl(x, y, dum):
    """Influence function for cov(x, y | dum = 1), dum in {0, 1}."""
    x, y, dum = _vec(x), _vec(y), _vec(dum)
    mu_dum = np.mean(dum)
    mu_x_y_dum = np.mean(x * y * dum)
    mu_x_dum = np.mean(x * dum)
    mu_y_dum = np.mean(y * dum)
    infl = (
        (mu_dum * (x * y * dum - mu_x_y_dum)
         - mu_x_y_dum * (dum - mu_dum)) / mu_dum ** 2
        - mu_y_dum / mu_dum * (mu_dum * (x * dum - mu_x_dum)
                               - mu_x_dum * (dum - mu_dum)) / mu_dum ** 2
        - mu_x_dum / mu_dum * (mu_dum * (y * dum - mu_y_dum)
                               - mu_y_dum * (dum - mu_dum)) / mu_dum ** 2
    )
    return _col(infl)


def skew_infl(x):
    """Influence function for skewness."""
    x = _vec(x)

    mu_x3 = np.mean(x ** 3)
    mu_x2 = np.mean(x ** 2)
    mu_x = np.mean(x)
    F = mu_x3 - 3 * mu_x2 * mu_x + 2 * mu_x ** 3
    G = mu_x2 - mu_x ** 2

    psi_x = x - mu_x
    psi_x2 = x ** 2 - mu_x2
    psi_x3 = x ** 3 - mu_x3
    psi_F = psi_x3 - 3 * (psi_x2 * mu_x + mu_x2 * psi_x) + 6 * mu_x ** 2 * psi_x
    psi_G = psi_x2 - 2 * mu_x * psi_x

    return _col((psi_F * G ** 1.5 - 1.5 * F * G ** 0.5 * psi_G) / G ** 3)


def skew_cond_infl(x, dum):
    """Influence function for conditional skewness, dum in {0, 1}."""
    x, dum = _vec(x), _vec(dum)

    x_dum = x * dum
    x2_dum = x ** 2 * dum
    x3_dum = x ** 3 * dum

    mu_dum = np.mean(dum)
    mu_x3_dum = np.mean(x3_dum)
    mu_x2_dum = np.mean(x2_dum)
    mu_x_dum = np.mean(x_dum)
    mu_c_x3 = mu_x3_dum / mu_dum
    mu_c_x2 = mu_x2_dum / mu_dum
    mu_c_x = mu_x_dum / mu_dum
    F_dum = mu_c_x3 - 3 * mu_c_x2 * mu_c_x + 2 * mu_c_x ** 3
    G_dum = mu_c_x2 - mu_c_x ** 2

    psi_x_dum = ((x_dum - mu_x_dum) * mu_dum
                 - mu_x_dum * (dum - mu_dum)) / mu_dum ** 2
    psi_x2_dum = ((x2_dum - mu_x2_dum) * mu_dum
                  - mu_x2_dum * (dum - mu_dum)) / mu_dum ** 2
    psi_x3_dum = ((x3_dum - mu_x3_dum) * mu_dum
                  - mu_x3_dum * (dum - mu_dum)) / mu_dum ** 2
    psi_F_dum = (psi_x3_dum
                 - 3 * (psi_x2_dum * mu_x_dum + mu_x2_dum * psi_x_dum)
                 + 6 * mu_x_dum ** 2 * psi_x_dum)
    psi_G_dum = psi_x2_dum - 2 * mu_x_dum * psi_x_dum

    return _col((psi_F_dum * G_dum ** 1.5
                 - 1.5 * F_dum * G_dum ** 0.5 * psi_G_dum) / G_dum ** 3)
