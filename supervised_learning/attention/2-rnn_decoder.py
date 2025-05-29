#!/usr/bin/env python3
"""RNN Decoder with Self-Attention"""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNN Decoder class"""

    def __init__(self, vocab, embedding, units, batch):
        """Initialize the RNNDecoder"""
        super(RNNDecoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )

        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

        self.F = tf.keras.layers.Dense(vocab)

        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """Forward pass"""
        # Compute context vector from attention mechanism
        context, _ = self.attention(s_prev, hidden_states)  # (batch, units)

        # Embed input token
        x = self.embedding(x)  # (batch, 1, embedding)

        # Add time dimension to context vector
        context = tf.expand_dims(context, axis=1)  # (batch, 1, units)

        # Concatenate context and embedded input
        x = tf.concat([context, x], axis=-1)  # (batch, 1, units + embedding)

        # Pass through GRU
        output, state = self.gru(x, initial_state=s_prev)  # output: (batch, 1, units)

        # Remove time dimension from GRU output
        output = output[:, 0, :]  # (batch, units)

        # Final output logits
        y = self.F(output)  # (batch, vocab)

        return y, state
