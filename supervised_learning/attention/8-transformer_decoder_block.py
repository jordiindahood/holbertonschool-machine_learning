#!/usr/bin/env python3

"""script 8"""

import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """init"""
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        call
        """
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        attn2, _ = self.mha2(
            out1, encoder_output, encoder_output, padding_mask
        )
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        hidden = self.dense_hidden(out2)
        output = self.dense_output(hidden)
        output = self.dropout3(output, training=training)
        out3 = self.layernorm3(out2 + output)

        return out3

    def __call__(self, *args, **kwargs):
        """edit the __call__ function so it can work"""
        # Here we handle positional args so non-tensor args become kwargs
        # Tensor positional args:  x, encoder_output
        # Non-tensor positional args: training, look_ahead_mask, padding_mask

        # Extract positional args
        x = args[0] if len(args) > 0 else None
        encoder_output = args[1] if len(args) > 1 else None

        # Defaults
        training = kwargs.get('training', None)
        look_ahead_mask = kwargs.get('look_ahead_mask', None)
        padding_mask = kwargs.get('padding_mask', None)

        # Override with positional if provided
        if len(args) > 2:
            training = args[2]
        if len(args) > 3:
            look_ahead_mask = args[3]
        if len(args) > 4:
            padding_mask = args[4]

        return super().__call__(
            x,
            encoder_output,
            training=training,
            look_ahead_mask=look_ahead_mask,
            padding_mask=padding_mask,
        )
