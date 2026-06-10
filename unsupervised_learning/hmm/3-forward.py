#!/usr/bin/env python3
"""script 3"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden Markov model.
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

        F = np.zeros((N, T))

        F[:, 0] = Initial.T * Emission[:, Observation[0]]

        for t in range(1, T):
            for j in range(N):
                F[j, t] = (
                    np.sum(F[:, t - 1] * Transition[:, j])
                    * Emission[j, Observation[t]]
                )

        P = np.sum(F[:, -1])
        return P, F

    except Exception:
        return None, None
