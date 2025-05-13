#!/usr/bin/env python3

"""script 1"""

import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix from a covariance matrix
    """

    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    stddev = np.sqrt(np.diag(C))

    stddev_outer = np.outer(stddev, stddev)

    with np.errstate(divide='ignore', invalid='ignore'):
        corr = C / stddev_outer
        corr[np.isnan(corr)] = 0

    return corr
