#!/usr/bin/env python3
"""exponential"""


class Exponential:
    """Class to represent an exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Initialize"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            self.lambtha = 1 / mean

        self.e = 2.7182818285

    def pdf(self, x):
        """Calculates the PDF for a given time period x"""
        if x < 0:
            return 0
        return self.lambtha * self.e**(-self.lambtha * x)
