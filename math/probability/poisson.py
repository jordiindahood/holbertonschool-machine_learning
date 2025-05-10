#!/usr/bin/env python3

"""script POISSON"""

import math

class Poisson:
    """script 0"""

    def __init__(self, data=None, lambtha=1.0):
        """Init"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculates the PMF for a given number of successes (k)"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        e = math.e
        λ = self.lambtha
        numerator = (e**-λ) * (λ**k)
        denominator = math.factorial(k)
        return numerator / denominator
