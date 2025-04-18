#!/usr/bin/env python3
""" script 14 """
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.
    """
    init = tf.keras.initializers.VarianceScaling(mode="fan_avg")
    layer = tf.keras.layers.Dense(n, kernel_initializer=init)
    z = layer(prev)
    gamma = tf.Variable(1.0, trainable=True)
    beta = tf.Variable(0.0, trainable=True)
    mean = tf.math.reduce_mean(z, axis=0)
    var = tf.math.reduce_variance(z, axis=0)
    epsilon = 1e-8
    normalized = tf.nn.batch_normalization(z, mean, var, beta, gamma, epsilon)
    return activation(normalized)
