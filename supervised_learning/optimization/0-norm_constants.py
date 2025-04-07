#!/usr/bin/env python3
""" Task 0"""
import numpy as np


def normalization_constants(X):
    """
     Calculates the mean and standard deviation of a given dataset X.
    """
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0)
    return m, s
