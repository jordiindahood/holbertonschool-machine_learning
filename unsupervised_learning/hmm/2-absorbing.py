#!/usr/bin/env python3
"""Absorbing Markov Chain"""

import numpy as np


def absorbing(P):
    """
    Determines if a Markov chain is absorbing.

    Parameters:
    - P: numpy.ndarray of shape (n, n), the transition matrix

    Returns:
    - True if the chain is absorbing, False otherwise
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return False
    n, m = P.shape
    if n != m:
        return False

    # Step 1: Identify absorbing states: P[i, i] == 1 and rest of row == 0
    absorbing_states = np.where(
        np.isclose(P.diagonal(), 1)
        & np.isclose(P - np.eye(n) * P.diagonal()[:, None], 0).all(axis=1)
    )[0]

    if len(absorbing_states) == 0:
        return False

    # Step 2: Check if all other states can reach an absorbing state
    # Construct reachability matrix by repeatedly multiplying P
    reachable = np.copy(P)
    for _ in range(n):
        reachable = np.matmul(reachable, P)

    for i in range(n):
        if i in absorbing_states:
            continue
        if not np.any(reachable[i, absorbing_states] > 0):
            return False

    return True
