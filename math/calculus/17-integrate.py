#!/usr/bin/env python3
"""
    matisse
"""


def poly_integral(poly, C=0):
    """
    calculation
    """
    if not isinstance(poly, list) or poly == []:
        return None

    if not isinstance(C, (int, float)):
        return None

    dx = [C]

    if poly == [0]:
        return dx

    for pow, coef in enumerate(poly):
        new_coef = coef / (pow + 1)
        
        dx.append(int(new_coef) if new_coef.is_integer() else new_coef)

        while len(dx) > 1 and dx[-1] == 0:
            dx.pop()

    return dx
