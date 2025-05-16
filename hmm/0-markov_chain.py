#!/usr/bin/env python3
"""Markov Chain Module"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a Markov chain being in a particular state
    after a specified number of iterations.
    """

    if not isinstance(P, np.ndarray) or not isinstance(s, np.ndarray):
        return None
    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return None
    n = P.shape[0]
    if s.shape != (1, n):
        return None
    if not isinstance(t, int) or t < 1:
        return None

    try:
        result = s @ np.linalg.matrix_power(P, t)
        return result
    except Exception:
        return None
