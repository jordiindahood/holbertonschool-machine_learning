#!/usr/bin/env python3
"""binomial distribution"""


class Binomial:
    """Class to represent a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Init"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            p = 1 - (variance / mean)
            n = round(mean / p)
            p = mean / n

            self.n = n
            self.p = p

    def pmf(self, k):
        """Calculates the PMF for a given number of successes (k)"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0 or k > self.n:
            return 0

        n_fact = self.factorial(self.n)
        k_fact = self.factorial(k)
        nk_fact = self.factorial(self.n - k)
        comb = n_fact / (k_fact * nk_fact)

        return comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def factorial(self, num):
        """Computes factorial of a number"""
        if num < 2:
            return 1
        result = 1
        for i in range(2, num + 1):
            result *= i
        return result
