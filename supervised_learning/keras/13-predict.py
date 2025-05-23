#!/usr/bin/env python3
""" tscript 13"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network.
    """
    return network.predict(data, verbose=verbose)
