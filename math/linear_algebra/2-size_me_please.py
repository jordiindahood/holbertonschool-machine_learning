#!/usr/bin/env python3


def matrix_shape(matrix):
    shape = []
    while matrix:
        try:
            shape.append(len(matrix))
            matrix = matrix[0]
        except Exception:
            return shape
