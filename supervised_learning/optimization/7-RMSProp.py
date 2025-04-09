#!/usr/bin/env python3
""" script 7 """


def update_variables_RMSProp(alpha, beta2, epsilon,
                             var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.
    """

    s_new = beta2 * s + (1 - beta2) * (grad * grad)
    W = var - alpha * (grad / ((s_new ** 0.5) + epsilon))

    return W, s_new
