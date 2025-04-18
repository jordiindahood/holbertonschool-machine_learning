#!/usr/bin/env python3
""" script 0 """
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.
    """
    summation = 0
    for idx in range(1, L + 1):
        key = "W{}".format(idx)
        summation += np.linalg.norm(weights[key])

    L2_cost = lambtha * summation / (2 * m)

    return cost + L2_cost
