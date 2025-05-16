#!/usr/bin/env python3
"""script 9"""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using BIC.
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    n, d = X.shape
    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None
    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax < kmin:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    log = []
    b = []
    results = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose
        )
        if (
            pi is None
            or m is None
            or S is None
            or g is None
            or log_likelihood is None
        ):
            return None, None, None, None

        l.append(log_likelihood)

        p = (k * d) + (k * d * (d + 1) / 2) + (k - 1)
        BIC_value = p * np.log(n) - 2 * log_likelihood
        b.append(BIC_value)

        results.append((pi, m, S))

    log = np.array(log)
    b = np.array(b)

    best_index = np.argmin(b)
    best_k = kmin + best_index
    best_result = results[best_index]

    return best_k, best_result, log, b
