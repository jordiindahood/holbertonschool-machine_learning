#!/usr/bin/env python3
""" script 12 """

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network.
    """
    return network.evaluate(data, labels, verbose=verbose)
