#!/usr/bin/env python3
"""
module file for riding_bareback

"""


def mat_mul(mat1, mat2):
    """a function that performs matrix multiplication

    Keyword arguments:
    mat1 -- matrix
    mat2 -- matrix
    Return: matrix

    """

    rows1 = len(mat1)  # 3
    cols1 = len(mat1[0])  # 2
    rows2 = len(mat2)  # 2
    cols2 = len(mat2[0])  # 4

    if cols1 != rows2:
        return None

    result = [[0 for idx in range(cols2)] for idx in range(rows1)]

    for i in range(rows1):
        for j in range(cols2):
            for k in range(cols1):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
