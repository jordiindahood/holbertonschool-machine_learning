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
        self.d = d

    def pdf(self, x):
        """
        Calculates the PDF at a data point
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if x.shape != (self.d, 1):
            raise ValueError(f"x must have the shape ({self.d}, 1)")

        x_m = x - self.mean

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)

        denom = np.sqrt((2 * np.pi) ** self.d * det)

        expo = -0.5 * (x_m.T @ inv @ x_m)

        return float((1 / denom) * np.exp(expo))
