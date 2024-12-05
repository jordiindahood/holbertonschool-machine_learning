#!/usr/bin/env python3
"""
module file for slice like a ninja

"""


def np_slice(matrix, axes={}):
    """a function that slices a matrix along specific axes

    Keyword arguments:
    mat1 -- numpy.ndarray
    axes -- dict
    Return -- numpy.ndarray
    """
    slices = [slice(None)] * matrix.ndim  # type: ignore

    for ky, vl in axes.items():
        slices[ky] = slice(*vl)

    return matrix[tuple(slices)]
