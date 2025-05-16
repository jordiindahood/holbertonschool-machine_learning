#!/usr/bin/env python3
"""script 12"""

import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on a dataset with Ward linkage.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(dist, (int, float)) or dist <= 0:
        return None

    Z = sch.linkage(X, method='ward')

    plt.figure()
    sch.dendrogram(Z, color_threshold=dist)
    plt.axhline(y=dist, c='k', ls='--')
    plt.title('Agglomerative Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()

    clss = sch.fcluster(Z, t=dist, criterion='distance') - 1
    return clss
