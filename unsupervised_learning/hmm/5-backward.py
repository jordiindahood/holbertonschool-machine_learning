#!/usr/bin/env python3
"""script 5"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden Markov model.
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

        B = np.zeros((N, T))
        B[:, T - 1] = (
            1  # At the last time step, the probability is 1 for all states
        )

        for t in reversed(range(T - 1)):
            for i in range(N):
                B[i, t] = np.sum(
                    Transition[i, :]
                    * Emission[:, Observation[t + 1]]
                    * B[:, t + 1]
                )

        P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])

        return P, B

    except Exception:
        return None, None
