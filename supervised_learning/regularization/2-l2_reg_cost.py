#!/usr/bin/env python3
""" script 2 """
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the total cost of a neural network including L2 regularization.
    """

    myList = list()
    for layer in model.layers:
        myList.append(tf.reduce_sum(layer.losses) + cost)

    return tf.stack(myList[1:])
