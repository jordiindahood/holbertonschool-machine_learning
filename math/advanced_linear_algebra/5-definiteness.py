#!/usr/bin/env python3
""" Task 5: 5. Definiteness """

import numpy as np


def definiteness(matrix):
    """
    Determines the definiteness of a square matrix.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError('matrix must be a numpy.ndarray')

    le = matrix.shape[0]
    if len(matrix.shape) != 2 or le != matrix.shape[1]:
        return None

    transpose = np.transpose(matrix)
    if not np.array_equal(transpose, matrix):
        return None

    w, v = np.linalg.eig(matrix)

    if all(w > 0):
        return 'Positive definite'
    elif all(w >= 0):
        return 'Positive semi-definite'
    elif all(w < 0):
        return 'Negative definite'
    elif all(w <= 0):
        return 'Negative semi-definite'
    else:
        return 'Indefinite'
