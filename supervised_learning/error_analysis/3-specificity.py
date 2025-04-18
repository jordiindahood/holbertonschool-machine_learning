#!/usr/bin/env python3
""" script 3 """
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.
    """
    fal_pos = confusion.sum(axis=0) - np.diag(confusion)
    fal_neg = confusion.sum(axis=1) - np.diag(confusion)
    true_pos = np.diag(confusion)
    true_neg = confusion.sum() - (fal_pos + fal_neg + true_pos)

    return (true_neg / (true_neg + fal_pos))
