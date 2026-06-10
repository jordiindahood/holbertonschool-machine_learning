#!/usr/bin/env python3
"""sccript 0"""


def determinant(matrix):
    """
    Computes the determinant of a square matrix using recursion.
    """

    if not isinstance(matrix, list) or not all(
        isinstance(row, list) for row in matrix
    ):
        raise TypeError("matrix must be a list of lists")

    if len(matrix[0]) == 0:
        return 1

    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0] if matrix[0] else 1

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    return sum(
        matrix[0][i]
        * (-1) ** i
        * determinant([row[:i] + row[i + 1:] for row in matrix[1:]])
        for i in range(len(matrix))
    )
