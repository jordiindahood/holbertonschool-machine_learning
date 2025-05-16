#!/usr/bin/env python3
"""script 10"""

import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means clustering on a dataset.
    """
    model = sklearn.cluster.KMeans(n_clusters=k, n_init='auto')
    model.fit(X)

    C = model.cluster_centers_
    idx = model.labels_

    return C, idx
