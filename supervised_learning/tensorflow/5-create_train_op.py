#!/usr/bin/env python3
""" Task 5"""
import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_eager_execution()


def create_train_op(loss, alpha):
    """
    creates the training operation for the network:

    """
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
