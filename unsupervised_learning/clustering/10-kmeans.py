#!/usr/bin/env python3
"""script 10"""
import numpy as np
from sklearn.cluster import KMeans


def kmeans(X, k):
    """
    Performs K-means clustering on a dataset.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None

    model = KMeans(n_clusters=k, n_init='auto')
    model.fit(X)

    C = model.cluster_centers_
    idx = model.labels_

    return C, idx
