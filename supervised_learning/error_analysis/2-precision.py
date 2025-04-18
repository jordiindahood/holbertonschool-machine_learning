#!/usr/bin/env python3
""" script 2 """
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.
    """
    precision = []
    i = 0
    for row in confusion:
        positive = row[i]
        column = confusion.sum(axis=0)
        precision.append(positive / column[i])
        i = i + 1

    return np.array(precision)
