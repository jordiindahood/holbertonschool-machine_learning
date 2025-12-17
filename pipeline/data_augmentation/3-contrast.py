#!/usr/bin/env python3
""" script 1 """
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Performs a random crop of an image.
    """
    return tf.image.random_contrast(image, lower, upper)
