#!/usr/bin/env python3
"""script 0"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder
    """
    # ====================
    # Build Encoder
    # ====================
    inputs = keras.Input(shape=(input_dims,))
    encoded = inputs
    for units in hidden_layers:
        encoded = keras.layers.Dense(units, activation='relu')(encoded)
    latent = keras.layers.Dense(latent_dims, activation='relu')(encoded)

    encoder = keras.Model(inputs, latent, name="encoder")

    # ====================
    # Build Decoder
    # ====================
    decoder_input = keras.Input(shape=(latent_dims,))
    decoded = decoder_input
    for units in reversed(hidden_layers):
        decoded = keras.layers.Dense(units, activation='relu')(decoded)
    output = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)

    decoder = keras.Model(decoder_input, output, name="decoder")

    # ====================
    # Build Autoencoder
    # ====================

    auto_input = inputs
    auto_output = decoder(encoder(auto_input))
    auto = keras.Model(auto_input, auto_output, name="autoencoder")

    # Compile the model
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
