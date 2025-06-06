#!/usr/bin/env python3

"""script 1"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    SelfAttention Class
    """

    def __init__(self, units):
        """
        init
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        call
        """
        s_prev_expanded = tf.expand_dims(s_prev, 1)

        score = self.V(
            tf.nn.tanh(self.W(s_prev_expanded) + self.U(hidden_states))
        )

        attention_weights = tf.nn.softmax(score, axis=1)

        context = tf.reduce_sum(attention_weights * hidden_states, axis=1)

        return context, attention_weights
