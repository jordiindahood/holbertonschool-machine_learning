#!/usr/bin/env python3

"""script 7"""

import numpy as np


def maximization(X, g):
    """Performs the maximization step in the EM algorithm for a GMM"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(g) is not np.ndarray or len(g.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    if n != n_g:
        return None, None, None

    try:
        N_k = np.sum(g, axis=1)

        pi = N_k / n

        m = (g @ X) / N_k[:, np.newaxis]

        S = np.empty((k, d, d))
        for i in range(k):
            X_centered = X - m[i]
            weighted = g[i][:, np.newaxis] * X_centered
            S[i] = (weighted.T @ X_centered) / N_k[i]

        return pi, m, S
    except Exception:
        return None, None, None
