#!/usr/bin/env python3
"""
Convolutional GAN Generator and Discriminator
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def convolutional_GenDiscr():
    """
    Builds and returns a convolutional Generator and
    Discriminator for 16x16 grayscale faces.

    Generator:
        - Input shape: (16,)
        - Dense → Reshape → UpSampling + Conv2D → BatchNorm → Activation
        - Output shape: (16,16,1)
        - Activation: tanh

    Discriminator:
        - Input shape: (16,16,1)
        - Conv2D → MaxPooling → Activation
        - Flatten → Dense
        - Output: 1
        - Activation: tanh

    Returns:
        generator (keras.Model): The generator model
        discriminator (keras.Model): The discriminator model
    """

    def get_generator():
        """
        Builds the generator model.

        Returns:
            keras.Model: Generator model
        """
        input_layer = keras.Input(shape=(16,), name="input_generator")

        # Dense projection
        x = layers.Dense(2048, activation="tanh")(input_layer)
        x = layers.Reshape((2, 2, 512))(x)

        # Upsampling block 1
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(64, kernel_size=3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("tanh")(x)

        # Upsampling block 2
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(16, kernel_size=3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("tanh")(x)

        # Upsampling block 3
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(1, kernel_size=3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        output_layer = layers.Activation("tanh")(x)

        generator = keras.Model(
            inputs=input_layer, outputs=output_layer, name="generator"
        )
        return generator

    def get_discriminator():
        """
        Builds the discriminator model.

        Returns:
            keras.Model: Discriminator model
        """
        input_layer = keras.Input(
            shape=(16, 16, 1), name="input_discriminator"
        )

        # Conv block 1
        x = layers.Conv2D(32, kernel_size=3, padding="same")(input_layer)
        x = layers.MaxPooling2D()(x)
        x = layers.Activation("tanh")(x)

        # Conv block 2
        x = layers.Conv2D(64, kernel_size=3, padding="same")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Activation("tanh")(x)

        # Conv block 3
        x = layers.Conv2D(128, kernel_size=3, padding="same")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Activation("tanh")(x)

        # Conv block 4
        x = layers.Conv2D(256, kernel_size=3, padding="same")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Activation("tanh")(x)

        # Dense output
        x = layers.Flatten()(x)
        output_layer = layers.Dense(1)(x)

        discriminator = keras.Model(
            inputs=input_layer, outputs=output_layer, name="discriminator"
        )
        return discriminator

    return get_generator(), get_discriminator()


# Example usage:
gen, discr = convolutional_GenDiscr()
print(gen.summary(line_length=100))
print(discr.summary(line_length=100))
