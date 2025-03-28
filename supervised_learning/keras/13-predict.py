#!/usr/bin/env python3
""" tscript 13"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network.

    Args:
        network (keras.Model): The neural network model to use for prediction.
        data (numpy.ndarray): The input data to make predictions on.
        verbose (bool): Whether to print output during the prediction process.

    Returns:
    """
    return network.predict(data, verbose=verbose)