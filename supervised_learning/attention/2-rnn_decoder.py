#!/usr/bin/env python3
"""RNN Decoder with attention"""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Decoder for sequence-to-sequence with attention"""

    def __init__(self, vocab, embedding, units, batch):
        """Constructor for RNNDecoder"""
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
            recurrent_initializer='glorot_uniform'
        )

        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """Performs the forward pass"""
        # Apply attention
        context, _ = self.attention(s_prev, hidden_states)  # (batch, units)

        # Embed the input word
        x = self.embedding(x)  # (batch, 1, embedding)

        # Concatenate context vector with input
        context = tf.expand_dims(context, 1)  # (batch, 1, units)
        x = tf.concat([context, x], axis=-1)  # (batch, 1, units + embedding)

        # Pass through GRU
        output, state = self.gru(x, initial_state=s_prev)  # output: (batch, 1, units)

        # Project output to vocabulary
        output = tf.reshape(output, (-1, output.shape[2]))  # (batch, units)
        y = self.F(output)  # (batch, vocab)

        return y, state
