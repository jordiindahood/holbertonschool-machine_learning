#!/usr/bin/env python3
import tensorflow as tf
import numpy as np

positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1
    ):
        super().__init__()
        self.N = N
        self.dm = dm

        # Embedding layer to convert input token indices to vectors of dim dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)

        # Precompute positional encoding for max_seq_len tokens
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        # Create N EncoderBlocks
        self.blocks = [
            EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]

        # Dropout layer to apply after adding positional encoding
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # Embed the input tokens (shape: batch, seq_len, dm)
        x = self.embedding(x)  # (batch, seq_len, dm)

        # Add positional encoding (slice to seq_len)
        x += self.positional_encoding[:seq_len, :]

        # Apply dropout
        x = self.dropout(x, training=training)

        # Pass through all EncoderBlocks sequentially
        for i in range(self.N):
            x = self.blocks[i](x, training=training, mask=mask)

        return x

    # Override __call__ to accept positional args (x, training, mask)
    def __call__(self, *args, **kwargs):
        # Accept positional args and assign to proper keywords
        if len(args) == 1:
            # Just x
            return super().__call__(args[0], **kwargs)
        elif len(args) == 2:
            # x, training
            return super().__call__(args[0], training=args[1], **kwargs)
        elif len(args) == 3:
            # x, training, mask
            return super().__call__(args[0], training=args[1], mask=args[2], **kwargs)
        else:
            # fallback to normal
            return super().__call__(*args, **kwargs)