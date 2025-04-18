#!/usr/bin/env python3
""" script 0000 """
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix by computing the dot
    product of the labels and logits.
    """
    return np.dot(labels.T, logits)
