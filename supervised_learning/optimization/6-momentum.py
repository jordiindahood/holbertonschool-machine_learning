#!/usr/bin/env python3
""" script 6  """
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Creates the training operation for a neural network in TensorFlow
    using the gradient descent with momentum optimization algorithm.
    """

    optimizer = tf.compat.v1.train.MomentumOptimizer(alpha, beta1)
    return optimizer
