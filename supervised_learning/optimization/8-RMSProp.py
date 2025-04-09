#!/usr/bin/env python3
""" script 8 """
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Creates the RMSProp optimization operation for a neural network
    using TensorFlow's Keras optimizer.
    """
    opt = tf.keras.optimizers.RMSprop(learning_rate=alpha,
                                      rho=beta2, epsilon=epsilon)
    return opt
