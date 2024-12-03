#!/usr/bin/env python3
"""
module file for getting_cozy

"""


def cat_matrices2D(mat1, mat2, axis=0):
    """a function that concats two matrices

    Keyword arguments:
    mat1 -- matrix
    mat2 -- matrix
    Return: matrix

    """
    new_mat = []

    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        for idx in range(len(mat1)):
            new_mat.append(mat1[idx] + mat2[idx])

    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        new_mat = [i[:] for i in mat1] + [i[:] for i in mat2]
    return new_mat
