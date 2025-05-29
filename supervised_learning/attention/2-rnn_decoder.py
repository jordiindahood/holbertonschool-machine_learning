#!/usr/bin/env python3

"""script 2"""

import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """rnn decoder class"""

    def __init__(self, vocab, embedding, units, batch):
        """init"""
        super(RNNDecoder, self).__init__()
        self.units = units
        self.batch = batch

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab, output_dim=embedding
        )

        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform',
        )

        self.F = tf.keras.layers.Dense(vocab)

        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """call"""
        context, _ = self.attention(s_prev, hidden_states)

        x = self.embedding(x)
        context = tf.expand_dims(context, 1)
        x = tf.concat([context, x], axis=-1)
        output, state = self.gru(x, initial_state=s_prev)

        y = self.F(tf.squeeze(output, axis=1))

        return y, state
