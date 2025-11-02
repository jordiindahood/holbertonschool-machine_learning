#!/usr/bin/env python3
"""
Initialize the Q-table for a FrozenLake environment
"""

import numpy as np


def q_init(env):
    """
    Initializes the Q-table.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    return Q
