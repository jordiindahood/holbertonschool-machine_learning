#!/usr/bin/env python3
"""
module file for across_the_planes

"""


def add_matrices2D(mat1, mat2):
    """a function that adds two matrices

    Keyword arguments:
    mat1 -- matrices
    mat2 -- matrices
    Return: list
    """
    result = []
    row = []

    if len(mat1) != len(mat2):
        return None

    for i in range(len(mat1)):
        arr1 = mat1[i]
        arr2 = mat2[i]
        if len(arr1) != len(arr2):
            return None

        for idx in range(len(arr1)):
            row.append(arr1[idx] + arr2[idx])
        result.append(row)
        row = []

    return result
