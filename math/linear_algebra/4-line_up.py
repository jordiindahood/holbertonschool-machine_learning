#!/usr/bin/env python3
"""
module file for add_arrays

"""


def add_arrays(arr1, arr2):
    """a function that returns the sum of two lists

    Keyword arguments:
    arr1 -- list of int/float
    arr2 -- list of int/float
    Return: list
    """
    if len(arr1) != len(arr2):
        return None

    result = []

    for idx in range(len(arr1)):
        result.append(arr1[idx] + arr2[idx])

    return result
