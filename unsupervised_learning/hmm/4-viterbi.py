#!/usr/bin/env python3
"""script 4"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Performs the Viterbi algorithm for a hidden Markov model.
    """
    try:
        T = Observation.shape[0]
        N, M = Emission.shape

        if (
            Transition.shape != (N, N)
            or Initial.shape != (N, 1)
            or not isinstance(Observation, np.ndarray)
        ):
            return None, None

        V = np.zeros((N, T))
        B = np.zeros((N, T), int)

        # Initialization
        V[:, 0] = Initial.T * Emission[:, Observation[0]]

        # Recursion
        for t in range(1, T):
            for j in range(N):
                prob = V[:, t - 1] * Transition[:, j]
                B[j, t] = np.argmax(prob)
                V[j, t] = np.max(prob) * Emission[j, Observation[t]]

        # Termination
        P = np.max(V[:, T - 1])
        last_state = np.argmax(V[:, T - 1])

        # Backtracking
        path = [last_state]
        for t in range(T - 1, 0, -1):
            last_state = B[last_state, t]
            path.insert(0, last_state)

        return path, P

    except Exception:
        return None, None
