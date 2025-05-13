#!/usr/bin/env python3
"""script 0"""

import numpy as np


def pca(X, var=0.95):
    """
    Performs Principal Component Analysis (PCA) on the given dataset.
    """
    u, s, vh = np.linalg.svd(X)

    cumsum = np.cumsum(s)

    dim = [i for i in range(len(s)) if cumsum[i] / cumsum[-1] >= var]
    ndim = dim[0] + 1

    return vh.T[:, :ndim]
