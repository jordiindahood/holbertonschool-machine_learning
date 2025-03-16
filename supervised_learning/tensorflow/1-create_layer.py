#!/usr/bin/env python3
""" Task 1 """
import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_eager_execution()


def create_layer(prev, n, activation):
    """
    Creates and returns TensorFlow placeholders for input data and labels.
    """

    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init,
        name='layer'
    )

    output = layer(prev)

    return output
