#!/usr/bin/env python3
""" script 9 """


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable using the Adam optimization algorithm.
    """
    Vd = (beta1 * v) + ((1 - beta1) * grad)
    Sd = (beta2 * s) + ((1 - beta2) * grad * grad)

    Vd_ok = Vd / (1 - beta1**t)
    Sd_ok = Sd / (1 - beta2**t)

    w = var - alpha * (Vd_ok / ((Sd_ok ** (0.5)) + epsilon))
    return (w, Vd, Sd)
