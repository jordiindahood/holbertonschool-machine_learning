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
        σ2 = self.sigma_f**2
        l2 = self.l**2

        sqr_sumx1 = np.sum(X1**2, 1).reshape(-1, 1)
        sqr_sumx2 = np.sum(X2**2, 1)
        sqr_dist = sqr_sumx1 - 2 * np.dot(X1, X2.T) + sqr_sumx2

        kernel = σ2 * np.exp(-0.5 / l2 * sqr_dist)
        return kernel

    def predict(self, X_s):
        """
        Predicts the mean and variance of points in a Gaussian Process.
        """
        s = X_s.shape[0]
        K = self.K
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s) + np.ones(s) - np.eye(s)
        K_inv = np.linalg.inv(K)

        μ = (K_s.T.dot(K_inv).dot(self.Y)).flatten()

        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        cov_s = np.diag(cov_s)

        return (μ, cov_s)
