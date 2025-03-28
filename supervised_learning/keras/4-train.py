#!/usr/bin/env python3
""" script 4"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent.
    """
    return network.fit(x=data,
                       y=labels,
                       epochs=epochs,
                       batch_size=batch_size,
                       shuffle=shuffle,
                       verbose=verbose)
