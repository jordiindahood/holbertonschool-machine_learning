#!/usr/bin/env python3
""" script 1 """
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity (true positive rate or recall) for each class
    from the confusion matrix.
    """
    sensitivity = []
    i = 0
    for row in confusion:
        positive = row[i]
        false = sum(row)
        sensitivity.append(positive / false)
        i = i + 1

    return np.array(sensitivity)
