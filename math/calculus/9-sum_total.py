#!/usr/bin/env python3
"""
    sum_total
"""


def summation_i_squared(n):
    """
    calculation
    """
    if type(n) == int and n > 0:
        return n * (n + 1) * (2 * n + 1) // 6
    return None