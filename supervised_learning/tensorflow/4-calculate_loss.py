#!/usr/bin/env python3
""" Task 4"""
import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_eager_execution()


def calculate_loss(y, y_pred):
    """
    calculate loss
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)
