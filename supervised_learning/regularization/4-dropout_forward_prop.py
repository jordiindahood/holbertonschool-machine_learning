#!/usr/bin/env python3
""" script 4 """
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Performs forward propagation with dropout regularization for
    a neural network.
    """

    myDict = dict()
    myDict['A0'] = X

    for idx in range(1, L + 1):
        keyA = "A{}".format(idx)
        keyA_p = "A{}".format(idx - 1)
        keyD = "D{}".format(idx)
        keyW = "W{}".format(idx)
        keyb = "b{}".format(idx)

        Z = np.matmul(weights[keyW], myDict[keyA_p]) + weights[keyb]

        if idx != L:
            A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
            D = np.random.rand(A.shape[0], A.shape[1])
            D = np.where(D < keep_prob, 1, 0)
            myDict[keyD] = D
            A *= D
            A /= keep_prob
            myDict[keyA] = A
        else:
            t = np.exp(Z)
            A = t / np.sum(t, axis=0, keepdims=True)
            myDict[keyA] = A

    return myDict
