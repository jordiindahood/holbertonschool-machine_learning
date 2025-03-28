#!/usr/bin/env python3
""" script 3"""
import tensorflow.keras as K  # type: ignore


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot encoded
    matrix using Keras utilities.
    """
    return K.utils.to_categorical(labels, classes)
