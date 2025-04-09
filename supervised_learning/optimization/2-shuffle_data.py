#!/usr/bin/env python3
""" script 2"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices in the same way
    """
    arr = X.shape[0]
    shuffle = np.random.permutation(arr)
    return X[shuffle], Y[shuffle]
