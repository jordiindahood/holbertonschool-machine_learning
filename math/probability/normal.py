#!/usr/bin/env python3
"""normal distribution"""


class Normal:
    """Class to represent a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Init"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = variance ** 0.5

        self.e = 2.7182818285
        self.pi = 3.1415926536

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        coeff = 1 / (self.stddev * (2 * self.pi) ** 0.5)
        exponent = -((x - self.mean) ** 2) / (2 * self.stddev ** 2)
        return coeff * self.e ** exponent

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        z = (x - self.mean) / (self.stddev * 2 ** 0.5)
        # Taylor approximation of erf(z)
        erf = (2 / (self.pi ** 0.5)) * (
            z - (z ** 3) / 3 + (z ** 5) / 10 -
            (z ** 7) / 42 + (z ** 9) / 216
        )
        return 0.5 * (1 + erf)
