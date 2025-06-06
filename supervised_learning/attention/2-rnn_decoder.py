#!/usr/bin/env python3
"""script 2"""

import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    Custom RNN decoder with attention mechanism.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Initializes the decoder.
        """
        super(RNNDecoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(vocab, embedding)

        self.gru = tf.keras.layers.GRU(
            units,
            recurrent_initializer='glorot_uniform',
            return_sequences=True,
            return_state=True,
        )
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Performs the forward pass of the decoder.
        """
        attention = SelfAttention(s_prev.shape[1])
        context, weights = attention(s_prev, hidden_states)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        outputs, s = self.gru(x)
        outputs = tf.reshape(outputs, (-1, outputs.shape[2]))
        y = self.F(outputs)

        return y, s
