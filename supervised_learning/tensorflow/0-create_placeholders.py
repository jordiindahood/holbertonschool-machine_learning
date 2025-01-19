#!/usr/bin/env python3
""" Task 0: 0. Placeholders"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Creates and returns TensorFlow placeholders for input data and labels.
    """
    x = tf.compat.v1.placeholder(float, shape=[None, nx], name="x")
    y = tf.compat.v1.placeholder(float, shape=[None, classes], name="y")

    return (x, y)
