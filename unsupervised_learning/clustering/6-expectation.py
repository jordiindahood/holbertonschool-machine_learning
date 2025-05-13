#!/usr/bin/env python3

"""script 6"""

import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Performs the expectation step in the EM algorithm for a GMM
    """
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or not isinstance(pi, np.ndarray)
        or pi.ndim != 1
        or not isinstance(m, np.ndarray)
        or m.ndim != 2
        or not isinstance(S, np.ndarray)
        or S.ndim != 3
    ):
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape != (k, d) or S.shape != (k, d, d) or pi.shape[0] != k:
        return None, None

    try:
        g = np.array([pi[i] * pdf(X, m[i], S[i]) for i in range(k)])

        likelihood = np.sum(g, axis=0)
        g /= likelihood

        log_likelihood = np.sum(np.log(likelihood))

        return g, log_likelihood
    except Exception:
        return None, None
