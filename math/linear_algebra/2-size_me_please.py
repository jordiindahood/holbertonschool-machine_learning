#!/usr/bin/env python3
"""
module file for matrix_shape

"""


def matrix_shape(matrix):
    """a function taht returns the shape of a matrix

    Keyword arguments:
    matrix -- ndarray
    Return: list of integers
    """

    shape = []
    while matrix:
        try:
            shape.append(len(matrix))
            matrix = matrix[0]
        except Exception:
            return shape
