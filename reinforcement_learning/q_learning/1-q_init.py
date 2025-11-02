#!/usr/bin/env python3
"""
Initialize the Q-table for a FrozenLake environment
"""

import numpy as np


def q_init(env):
    """
    Initializes the Q-table.

    Args:
        env: the FrozenLakeEnv instance

    Returns:
        A numpy.ndarray of zeros with shape (number_of_states, number_of_actions)
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    return Q
