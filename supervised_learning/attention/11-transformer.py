#!/usr/bin/env python3

"""script 11"""

import tensorflow as tf

Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """transformer class"""

    def __init__(
        self,
        N,
        dm,
        h,
        hidden,
        input_vocab,
        target_vocab,
        max_seq_input,
        max_seq_target,
        drop_rate=0.1,
    ):
        """init"""
        super().__init__()
        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_seq_input, drop_rate
        )
        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab, max_seq_target, drop_rate
        )
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(
        self,
        inputs,
        target,
        training,
        encoder_mask,
        look_ahead_mask,
        decoder_mask,
    ):
        """call func"""
        enc_output = self.encoder(inputs, training=training, mask=encoder_mask)

        dec_output = self.decoder(
            target,
            enc_output,
            training=training,
            look_ahead_mask=look_ahead_mask,
            padding_mask=decoder_mask,
        )

        final_output = self.linear(dec_output)
        return final_output

    def __call__(self, *args, **kwargs):
        """overrinding __call__ so it can work"""
        # Fix to accept positional args in this order:
        # inputs, target, training, encoder_mask, look_ahead_mask, decoder_mask
        if len(args) > 0:
            kwargs['inputs'] = args[0]
        if len(args) > 1:
            kwargs['target'] = args[1]
        if len(args) > 2:
            kwargs['training'] = args[2]
        if len(args) > 3:
            kwargs['encoder_mask'] = args[3]
        if len(args) > 4:
            kwargs['look_ahead_mask'] = args[4]
        if len(args) > 5:
            kwargs['decoder_mask'] = args[5]

        return super().__call__(**kwargs)
