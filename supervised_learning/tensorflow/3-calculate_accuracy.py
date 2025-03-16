#!/usr/bin/env python3
""" Task 3: 3. Accuracy """
import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_eager_execution()


def calculate_accuracy(y, y_pred):
    """
    calculates the accuracy of predictions compared to the true labels.
    """

    correct_prediction = tf.equal(tf.argmax(y, 1),
                                  tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy
