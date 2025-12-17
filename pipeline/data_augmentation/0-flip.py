#!/usr/bin/env python3
""" script 0 """

import tensorflow as tf


def flip_image(image):
    """sccript 0"""
    return tf.image.flip_left_right(image)
