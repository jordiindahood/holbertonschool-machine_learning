#!/usr/bin/env python3
"""script 0"""

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if not (0 < var <= 1):
        raise ValueError("var must be a float between 0 and 1")
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    variances = (S**2) / (X.shape[0] - 1)
    total_var = np.sum(variances)
    explained = np.cumsum(variances) / total_var

    nd = np.searchsorted(explained, var) + 1

    W = Vt[:nd].T
    return W
