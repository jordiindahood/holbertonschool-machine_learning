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
