#!/usr/bin/env python3
"""script 2"""

import tensorflow as tf
from tensorflow.keras import layers, models


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder
    """
    inputs = layers.Input(shape=input_dims)
    x = inputs
    for f in filters:
        x = layers.Conv2D(
            filters=f, kernel_size=(3, 3), padding='same', activation='relu'
        )(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    encoder = models.Model(inputs, x, name='encoder')

    latent_inputs = layers.Input(shape=latent_dims)
    x = latent_inputs

    for f in filters[:-1][::-1]:
        x = layers.Conv2D(
            filters=f, kernel_size=(3, 3), padding='same', activation='relu'
        )(x)
        x = layers.UpSampling2D(size=(2, 2))(x)

    x = layers.Conv2D(
        filters=filters[0],
        kernel_size=(3, 3),
        padding='valid',
        activation='relu',
    )(x)
    x = layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        padding='same',
        activation='sigmoid',
    )(x)

    decoder = models.Model(latent_inputs, x, name='decoder')

    auto_input = inputs
    encoded = encoder(auto_input)
    decoded = decoder(encoded)

    auto = models.Model(auto_input, decoded, name='autoencoder')
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
