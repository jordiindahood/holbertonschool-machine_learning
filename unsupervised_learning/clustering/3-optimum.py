#!/usr/bin/env python3

"""script 3"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Tests for the optimum number of clusters by variance"""

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    n, d = X.shape

    if not isinstance(kmin, int) or kmin < 1:
        return None, None
    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax < kmin:
        return None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None

    results = []
    d_vars = []

    variances = []

    for k in range(kmin, kmax + 1):
        centroids, clss = kmeans(X, k, iterations)
        results.append((centroids, clss))
        var = variance(X, centroids)
        variances.append(var)

    base_variance = variances[0]
    d_vars = [base_variance - v for v in variances]

    return results, d_vars
