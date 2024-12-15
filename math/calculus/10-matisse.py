#!/usr/bin/env python3
"""
    matisse
"""


def poly_derivative(poly):
    """
    calculation
    """
    if not isinstance(poly, list) or poly == []:
        return None

    if len(poly) == 1:
        return [0]

    dx = [coeff * poww for poww, coeff in enumerate(poly) if poww > 0]
    return dx
