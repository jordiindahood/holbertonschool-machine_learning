#!/usr/bin/env python3
""" Regular Markov Chain Steady State """

import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular Markov chain
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    n, m = P.shape
    if n != m:
        return None

    power = np.linalg.matrix_power(P, 1)
    for i in range(1, 100):
        power = np.linalg.matrix_power(P, i)
        if np.all(power > 0):
            break
    else:
        return None

    A = np.vstack((P.T - np.eye(n), np.ones((1, n))))
    b = np.zeros((n + 1,))
    b[-1] = 1

    try:
        steady_state = np.linalg.lstsq(A, b, rcond=None)[0]
        return steady_state[np.newaxis, :]
    except Exception:
        return None
