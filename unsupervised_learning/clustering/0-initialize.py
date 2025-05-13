#!/usr/bin/env python3
"""script"""

import numpy as np


def initialize(X, k):
    """
    Init
    """
    if (
        not isinstance(X, np.ndarray)
        or not isinstance(k, int)
        or X.ndim != 2
        or k <= 0
    ):
        return None
    n, d = X.shape
    centroids = np.zeros((k, X.shape[1]))
    centroids = np.random.uniform(low=X.min(axis=0), high=X.max(axis=0),
                                  size=(k, d))
    return centroids
