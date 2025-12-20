#!/usr/bin/env python3
"""
Neural Style Transfer class
"""

import numpy as np
import tensorflow as tf


class NST:
    """Neural Style Transfer class"""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1',
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize NST"""

        # Validate style_image
        if (
            not isinstance(style_image, np.ndarray)
            or style_image.ndim != 3
            or style_image.shape[2] != 3
        ):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        # Validate content_image
        if (
            not isinstance(content_image, np.ndarray)
            or content_image.ndim != 3
            or content_image.shape[2] != 3
        ):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        # Validate alpha
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        # Validate beta
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        # Set attributes
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

        # Load model
        self.load_model()

    @staticmethod
    def scale_image(image):
        """Rescale image"""

        if (
            not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape
        scale = 512 / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        image = tf.image.resize(
            image, (new_h, new_w), method=tf.image.ResizeMethod.BICUBIC
        )

        image = tf.clip_by_value(image / 255.0, 0.0, 1.0)
        image = tf.expand_dims(image, axis=0)

        return image

    def load_model(self):
        """Creates VGG19-based model with Average Pooling"""

        # Load pretrained VGG19
        vgg = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet'
        )
        vgg.trainable = False

        inputs = tf.keras.Input(shape=(None, None, 3))
        x = inputs
        outputs = []

        # Rebuild VGG19, replacing MaxPool with AvgPool
        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.Conv2D):
                x = tf.keras.layers.Conv2D(
                    filters=layer.filters,
                    kernel_size=layer.kernel_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    activation=layer.activation,
                    name=layer.name,
                    weights=layer.get_weights(),
                )(x)

            elif isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name.replace("max", "avg"),
                )(x)

            # Save style/content outputs
            if (
                layer.name in self.style_layers
                or layer.name == self.content_layer
            ):
                outputs.append(x)

            # Stop after content layer
            if layer.name == self.content_layer:
                break

        # Final model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.trainable = False
