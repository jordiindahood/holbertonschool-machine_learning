#!/usr/bin/env python3
"""
module file for cats_got_your_tongue

"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """a function that concats two matrices

    Keyword arguments:
    mat1 -- numpy.ndarray
    mat2 -- numpy.ndarray
    Return -- numpy.ndarray
    """

    return np.concatenate(mat1, mat2, axis=axis)
