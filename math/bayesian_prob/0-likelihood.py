#!/usr/bin/env python3

"""script 0"""

import numpy as np

def n_choose_x(n, x):
    """n_choose_x"""
    n_fact = np.math.factorial(n)
    x_fact = np.math.factorial(x)
    nx_fact = np.math.factorial(n - x)
    return n_fact / (x_fact * nx_fact)


def likelihood(x, n, P):
    """likelihood"""

    if (not isinstance(n, (int, float)) or n <= 0):
        raise ValueError("n must be a positive integer")

    if (not isinstance(x, (int, float)) or x < 0):
        message = "x must be an integer that is greater than or equal to 0"
        raise ValueError(message)

    if (x > n):
        raise ValueError("x cannot be greater than n")

    if (not isinstance(P, np.ndarray) or len(P.shape) != 1 or P.shape[0] < 1):
        raise TypeError("P must be a 1D numpy.ndarray")

    if (np.any(P > 1) or np.any(P < 0)):
        raise ValueError("All values in P must be in the range [0, 1]")

    binomial_coef = n_choose_x(n, x)
    success_rate = pow(P, x)
    failure_rate = pow(1 - P, n - x)

    likelihood = binomial_coef * success_rate * failure_rate

    return likelihood
