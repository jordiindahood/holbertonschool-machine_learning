#!/usr/bin/env python3
""" script 9 """
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model to a file.
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    Loads an entire model from a file.
    """
    return K.models.load_model(filename)
