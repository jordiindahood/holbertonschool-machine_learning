#!/usr/bin/env python3
""" script 1 """
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Performs a random crop of an image.
    """
    return tf.image.adjust_brightness(image, max_delta)
