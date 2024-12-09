#!/usr/bin/env python3
"""
module file for the_whole_barn

"""


def add_matrices(mat1, mat2):
    """a function that adds two matrices

    Keyword arguments:
    mat1 -- numpy.ndarray
    mat2 -- numpy.ndarray
    Return -- numpy.ndarray
    """

    # Check if the matrices have the same shape
    if len(mat1) != len(mat2) or any(len(row1) != len(row2) for row1, row2 in zip(mat1, mat2)):
        return None
    
    # Add the matrices element-wise
    result = [[a + b for a, b in zip(row1, row2)] for row1, row2 in zip(mat1, mat2)]
    return result
