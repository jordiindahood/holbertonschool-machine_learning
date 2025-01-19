#!/usr/bin/env python3
""" Task 0: 0. Placeholders"""
import tensorflow.compat.v1 as tf # type: ignore


def create_placeholders(nx, classes):
    """
    Creates and returns TensorFlow placeholders for input data and labels.
    """
    x = tf.placeholder(float, shape=[None, nx], name="x")
    y = tf.placeholder(float, shape=[None, classes], name="y")

    return (x, y)
