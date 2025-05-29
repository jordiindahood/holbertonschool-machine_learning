#!/usr/bin/env python3

"""script 10"""


import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """decoder class"""

    def __init__(
        self, N, dm, h, hidden, target_vocab, max_seq_len, drop_rate=0.1
    ):
        """init"""
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(
        self,
        x,
        encoder_output,
        training=False,
        look_ahead_mask=None,
        padding_mask=None,
    ):
        """call func"""
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len, :]
        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(
                x,
                encoder_output,
                training=training,
                look_ahead_mask=look_ahead_mask,
                padding_mask=padding_mask,
            )
        return x

    def __call__(
        self,
        x,
        encoder_output,
        training=False,
        look_ahead_mask=None,
        padding_mask=None,
    ):
        """overriding __call__"""
        # Override __call__ to allow positional arguments
        return super().__call__(
            x=x,
            encoder_output=encoder_output,
            training=training,
            look_ahead_mask=look_ahead_mask,
            padding_mask=padding_mask,
        )
