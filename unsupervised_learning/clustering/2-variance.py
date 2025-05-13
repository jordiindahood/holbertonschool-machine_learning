#!/usr/bin/env python3

"""script 2"""

import numpy as np


def variance(X, C):
    """Calculates the total intra-cluster variance"""
    if (
        not isinstance(X, np.ndarray)
        or len(X.shape) != 2
        or not isinstance(C, np.ndarray)
        or len(C.shape) != 2
        or X.shape[1] != C.shape[1]
    ):
        return None

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)

    var = np.sum(np.square(X - C[clss]))

    return var
