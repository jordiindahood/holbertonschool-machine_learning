#!/usr/bin/env python3
""" script 13 """


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network
    """

    X = Z.mean(0)
    alpha = Z.std(0) ** 2

    z_normalized = (Z - X) / ((alpha + epsilon) ** (0.5))
    result = gamma * z_normalized + beta

    return result
