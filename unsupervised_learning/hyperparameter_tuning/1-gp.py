#!/usr/bin/env python3
"""script 0"""

import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Constructor for the GaussianProcess class
        """

        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Computes the covariance kernel matrix using the RBF kernel
        """
        sqdist = (
            np.sum(X1**2, axis=1).reshape(-1, 1)
            + np.sum(X2**2, axis=1)
            - 2 * np.dot(X1, X2.T)
        )
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """
        Predicts the mean and variance of points in a Gaussian process
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = K_s.T.dot(K_inv).dot(self.Y).reshape(-1)

        cov = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sigma = np.diag(cov)

        return mu, sigma
