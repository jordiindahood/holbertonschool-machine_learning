#!/usr/bin/env python3
"""Multivariate Normal distribution"""

import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution
    """

    def __init__(self, data):
        """
        Init
        """

        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)

        X = data - self.mean

        self.cov = (X @ X.T) / (n - 1)
