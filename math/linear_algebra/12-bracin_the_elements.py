#!/usr/bin/env python3
"""
module file for bracin_the_elements

"""


def np_elementwise(mat1, mat2):
    """a function that performs element-wise addition,
    subtraction, multiplication, and division on two matrices.

    Keyword arguments:
    mat1 -- numpy.ndarray
    mat2 -- numpy.ndarray
    Return -- tuple
    """
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2

    return add, sub, mul, div
