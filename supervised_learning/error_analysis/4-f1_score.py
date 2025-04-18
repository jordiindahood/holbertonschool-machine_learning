#!/usr/bin/env python3
""" script 4 """
sensitivity = __import__("1-sensitivity").sensitivity
precision = __import__("2-precision").precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix.
    """
    prec = precision(confusion)
    sens = sensitivity(confusion)
    return 2 * (prec * sens) / (prec + sens)
