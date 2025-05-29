#!/usr/bin/env python3

"""script 9"""

import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """encoder Class"""

    def __init__(
        self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1
    ):
        """init"""
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training=False, mask=None):
        """call func"""
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len, :]
        x = self.dropout(x, training=training)
        for block in self.blocks:
            x = block(x, training=training, mask=mask)
        return x

    def __call__(self, *args, **kwargs):
        """overriding __call__ for positional arguments"""
        # Map positional args to call's keyword args
        if len(args) == 1:
            return super().__call__(args[0], **kwargs)
        elif len(args) == 2:
            return super().__call__(args[0], training=args[1], **kwargs)
        elif len(args) == 3:
            return super().__call__(
                args[0], training=args[1], mask=args[2], **kwargs
            )
        else:
            return super().__call__(*args, **kwargs)
