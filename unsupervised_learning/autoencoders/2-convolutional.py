#!/usr/bin/env python3
"""script 2"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder
    """
    input_encoder = keras.Input(shape=input_dims)
    input_decoder = keras.Input(shape=latent_dims)

    encoded = keras.layers.Conv2D(
        filters[0], kernel_size=(3, 3), padding='same', activation='relu'
    )(input_encoder)
    encoded = keras.layers.MaxPool2D((2, 2), padding='same')(encoded)
    for enc in range(1, len(filters)):
        encoded = keras.layers.Conv2D(
            filters[enc], kernel_size=(3, 3), padding='same', activation='relu'
        )(encoded)
        encoded = keras.layers.MaxPool2D((2, 2), padding='same')(encoded)

    latent = encoded
    encoder = keras.Model(inputs=input_encoder, outputs=latent)

    decoded = keras.layers.Conv2D(
        filters[-1], kernel_size=(3, 3), padding='same', activation='relu'
    )(input_decoder)
    decoded = keras.layers.UpSampling2D((2, 2))(decoded)

    for dec in range(len(filters) - 2, 0, -1):
        decoded = keras.layers.Conv2D(
            filters[dec], kernel_size=(3, 3), padding='same', activation='relu'
        )(decoded)
        decoded = keras.layers.UpSampling2D((2, 2))(decoded)

    last = keras.layers.Conv2D(
        filters[0], kernel_size=(3, 3), padding='valid', activation='relu'
    )(decoded)
    last = keras.layers.UpSampling2D((2, 2))(last)
    last = keras.layers.Conv2D(
        input_dims[-1],
        kernel_size=(3, 3),
        padding='same',
        activation='sigmoid',
    )(last)
    decoder = keras.Model(inputs=input_decoder, outputs=last)

    encoder_output = encoder(input_encoder)
    decoder_output = decoder(encoder_output)
    auto = keras.Model(inputs=input_encoder, outputs=decoder_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
