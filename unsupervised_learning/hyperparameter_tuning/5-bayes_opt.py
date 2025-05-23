#!/usr/bin/env python3
"""script 3"""

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
        """
        init
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Computes the next best sample location using the Expected
        Improvement (EI) acquisition function.
        """

        m_sample, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            sam = np.min(self.gp.Y)
            imp = sam - m_sample - self.xsi
        else:
            sam = np.max(self.gp.Y)
            imp = m_sample - sam - self.xsi

        with np.errstate(divide='warn'):
            Z = imp / sigma
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0

        X_nest = self.X_s[np.argmax(EI)]
        return X_nest, EI

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function using Bayesian Optimization.
        """
        X_opt = 0
        Y_opt = 0

        for _ in range(iterations):
            # Find the next best sample
            X_next = self.acquisition()[0]

            # if X_next already sampled in gp.X, ommit
            if X_next in self.gp.X:
                break

            else:
                # get Y_next, evaluate X_next is black box function
                Y_next = self.f(X_next)

                # updates a GP
                self.gp.update(X_next, Y_next)

                # if minimizing save the least otherwise save the largest
                if (Y_next < Y_opt) and (self.minimize):
                    X_opt, Y_opt = X_next, Y_next

                if not self.minimize and Y_next > Y_opt:
                    X_opt, Y_opt = X_next, Y_next

        # removing last element
        self.gp.X = self.gp.X[:-1]

        return X_opt, Y_opt
