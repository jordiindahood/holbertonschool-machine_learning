#!/usr/bin/env python3
"""script POISSON"""


class Poisson:
    """Represents a Poisson distribution"""

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
        self.pi = 3.1415926536
        self.e = 2.7182818285

    def pmf(self, k):
        """Calculates the PMF for a given number of successes (k)"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        e = self.e
        λ = self.lambtha
        numerator = (e**-λ) * (λ**k)
        denominator = self.factorial(k)
        return numerator / denominator

    def cdf(self, k):
        """Calculates the CDF for a given number of successes (k)"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        cumulative = 0
        for i in range(k + 1):
            cumulative += self.pmf(i)
        return cumulative

    def factorial(self, n):
        """Calculates factorial"""
        if n < 2:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result
