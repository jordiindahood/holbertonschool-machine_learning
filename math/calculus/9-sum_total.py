#!/usr/bin/env python3
"""
    sum_total
"""


def summation_i_squared(n):
    """
    calculation
    """
    result = 0
    for i in range(1, n + 1):
        result += i**2
    return result
