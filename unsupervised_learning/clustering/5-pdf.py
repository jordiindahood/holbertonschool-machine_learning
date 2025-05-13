#!/usr/bin/env python3

"""script 5"""

import numpy as np


def pdf(X, m, S):
    """Calculates the PDF of a multivariate Gaussian distribution"""
    if (
        not isinstance(X, np.ndarray)
        or len(X.shape) != 2
        or not isinstance(m, np.ndarray)
        or len(m.shape) != 1
        or not isinstance(S, np.ndarray)
        or len(S.shape) != 2
    ):
        return None

    n, d = X.shape
    if m.shape[0] != d or S.shape != (d, d):
        return None

    try:
        det = np.linalg.det(S)
        inv = np.linalg.inv(S)
        norm_const = 1.0 / (np.sqrt(((2 * np.pi) ** d) * det))

        diff = X - m
        exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)

        P = norm_const * np.exp(exponent)
        P = np.maximum(P, 1e-300)

        return P
    except Exception:
        return None
