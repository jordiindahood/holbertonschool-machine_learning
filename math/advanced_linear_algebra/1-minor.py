#!/usr/bin/env python3
"""script 1"""


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


def minor(matrix):
    """
    Computes the minor of a square matrix.
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError('matrix must be a list of lists')

    for element in matrix:
        if not isinstance(element, list):
            raise TypeError('matrix must be a list of lists')

    if len(matrix) == 1 and len(matrix[0]) == 0:
        raise ValueError('matrix must be a non-empty square matrix')

    for element in matrix:
        if len(element) != len(matrix):
            raise ValueError('matrix must be a non-empty square matrix')
    if len(matrix) == 1:
        return [[1]]

    minor = []
    for i in range(len(matrix)):
        minor.append([])
        for j in range(len(matrix)):
            rows = [matrix[m] for m in range(len(matrix)) if m != i]
            new_m = [
                [row[n] for n in range(len(matrix)) if n != j] for row in rows
            ]
            my_det = determinant(new_m)
            minor[i].append(my_det)

    return minor
