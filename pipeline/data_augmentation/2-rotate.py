#!/usr/bin/env python3
""" script 2 """
import tensorflow as tf


def rotate_image(image):
    """
    Performs a random crop of an image.
    """
    return tf.image.rot90(image, k=1)
