#!/usr/bin/env python3
"""
module file for matrix_transpose

"""


def matrix_transpose(matrix):
    """a function that returns the transpose of a matrix

    Keyword arguments:
    matrix -- ndarray
    Return: matrix
    """
    new_metrix = []
    row = []
    for idx in range(len(matrix[0])):
        for list in matrix:
            row.append(list[idx])
        new_metrix.append(row)
        row = []
    return new_metrix
