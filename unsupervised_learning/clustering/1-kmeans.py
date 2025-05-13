#!/usr/bin/env python3

"""script 1"""

import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means using uniform distribution"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    n, d = X.shape
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    return np.random.uniform(low=min_vals, high=max_vals, size=(k, d))


def kmeans(X, k, iterations=1000):
    """Performs K-means clustering on a dataset"""
    if (
        not isinstance(X, np.ndarray)
        or len(X.shape) != 2
        or not isinstance(k, int)
        or k <= 0
        or not isinstance(iterations, int)
        or iterations <= 0
    ):
        return None, None

    n, d = X.shape
    C = initialize(X, k)
    if C is None:
        return None, None

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        new_C = np.copy(C)
        for i in range(k):
            if np.any(clss == i):
                new_C[i] = np.mean(X[clss == i], axis=0)
            else:
                new_C[i] = np.random.uniform(
                    np.min(X, axis=0), np.max(X, axis=0)
                )

        if np.allclose(C, new_C):
            break
        C = new_C

    return C, clss
