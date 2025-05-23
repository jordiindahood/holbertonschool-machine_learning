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

        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices
        using the Radial Basis Function (RBF).
        """
        # K(xᵢ, xⱼ) = σ² exp((-0.5 / 2l²)(xᵢ − xⱼ)ᵀ (xᵢ − xⱼ))
        σ2 = self.sigma_f**2
        l2 = self.l**2

        sqr_sumx1 = np.sum(X1**2, 1).reshape(-1, 1)
        # print("sqr_sum1", sqr_sumx1)
        sqr_sumx2 = np.sum(X2**2, 1)
        # print("sqr_sum2", sqr_sumx2)
        sqr_dist = sqr_sumx1 - 2 * np.dot(X1, X2.T) + sqr_sumx2

        kernel = σ2 * np.exp(-0.5 / l2 * sqr_dist)
        return kernel
