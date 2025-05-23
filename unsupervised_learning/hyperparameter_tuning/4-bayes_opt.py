#!/usr/bin/env python3
"""script 4"""

import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process"""

    def __init__(
        self,
        f,
        X_init,
        Y_init,
        bounds,
        ac_samples,
        l=1,
        sigma_f=1,
        xsi=0.01,
        minimize=True,
    ):
        """init"""
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Computes the next best sample location using Expected Improvement (EI)
        """
        mu, sigma = self.gp.predict(self.X_s)
        sigma = np.sqrt(sigma)

        if self.minimize:
            best_Y = np.min(self.gp.Y)
            imp = best_Y - mu - self.xsi
        else:
            best_Y = np.max(self.gp.Y)
            imp = mu - best_Y - self.xsi

        with np.errstate(divide='warn'):
            Z = np.zeros_like(imp)
            mask = sigma > 0
            Z[mask] = imp[mask] / sigma[mask]
            EI = np.zeros_like(imp)
            EI[mask] = imp[mask] * norm.cdf(Z[mask]) + sigma[mask] * norm.pdf(
                Z[mask]
            )

        X_next = self.X_s[np.argmax(EI)].reshape(1)

        return X_next, EI
