#!/usr/bin/env python3
"""Baum-Welch algorithm for HMMs"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a hidden Markov model
    """
    try:
        T = Observations.shape[0]
        M, N = Emission.shape

        for _ in range(iterations):
            alpha = np.zeros((M, T))
            alpha[:, 0] = Initial[:, 0] * Emission[:, Observations[0]]
            for t in range(1, T):
                for j in range(M):
                    alpha[j, t] = (
                        np.sum(alpha[:, t - 1] * Transition[:, j])
                        * Emission[j, Observations[t]]
                    )

            beta = np.zeros((M, T))
            beta[:, T - 1] = 1
            for t in reversed(range(T - 1)):
                for i in range(M):
                    beta[i, t] = np.sum(
                        Transition[i, :]
                        * Emission[:, Observations[t + 1]]
                        * beta[:, t + 1]
                    )

            xi = np.zeros((M, M, T - 1))
            for t in range(T - 1):
                denom = np.dot(
                    np.dot(alpha[:, t].T, Transition)
                    * Emission[:, Observations[t + 1]].T,
                    beta[:, t + 1],
                )
                for i in range(M):
                    numer = (
                        alpha[i, t]
                        * Transition[i, :]
                        * Emission[:, Observations[t + 1]]
                        * beta[:, t + 1]
                    )
                    xi[i, :, t] = numer / denom

            gamma = np.sum(xi, axis=1)
            prod = alpha[:, T - 1] * beta[:, T - 1]
            gamma = np.hstack((gamma, prod[:, None] / np.sum(prod)))

            Transition = (
                np.sum(xi, axis=2) / np.sum(gamma[:, :-1], axis=1)[:, None]
            )

            for j in range(N):
                mask = Observations == j
                Emission[:, j] = np.sum(gamma[:, mask], axis=1)
            Emission /= np.sum(gamma, axis=1)[:, None]

        return Transition, Emission

    except Exception:
        return None, None
