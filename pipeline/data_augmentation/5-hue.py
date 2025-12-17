#!/usr/bin/env python3
""" script 1 """
import tensorflow as tf


def change_hue(image, delta):
    """
    Performs a random crop of an image.
    """
    return tf.image.adjust_hue(image, delta)
