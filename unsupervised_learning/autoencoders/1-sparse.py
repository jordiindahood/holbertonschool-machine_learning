#!/usr/bin/env python3
"""SCRIPT 1"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder
    """
    # ----- Encoder -----
    input_img = keras.Input(shape=(input_dims,))
    x = input_img
    for nodes in hidden_layers:
        x = keras.layers.Dense(units=nodes, activation='relu')(x)

    # Apply L1 regularization on the encoded representation
    encoded = keras.layers.Dense(
        units=latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha),
    )(x)
    encoder = keras.Model(inputs=input_img, outputs=encoded)

    # ----- Decoder -----
    latent_input = keras.Input(shape=(latent_dims,))
    x = latent_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(units=nodes, activation='relu')(x)
    decoded = keras.layers.Dense(units=input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=latent_input, outputs=decoded)

    # ----- Autoencoder -----
    auto_input = input_img
    auto_output = decoder(encoder(auto_input))
    auto = keras.Model(inputs=auto_input, outputs=auto_output)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
