#!/usr/bin/env python3
"""
Implements the epsilon-greedy action selection
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action.
    """
    p = np.random.uniform(0, 1)
    n_actions = Q.shape[1]

    if p < epsilon:
        action = np.random.randint(0, n_actions)
    else:
        action = np.argmax(Q[state])

    return action
