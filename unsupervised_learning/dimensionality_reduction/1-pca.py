#!/usr/bin/env python3
"""scritp 1"""

import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if not isinstance(ndim, int) or ndim <= 0 or ndim > X.shape[1]:
        raise ValueError(
            "ndim must be a positive integer less \
            than or equal to the number of features"
        )

    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    W = Vt[:ndim].T

    T = np.dot(X_centered, W)

    return T
