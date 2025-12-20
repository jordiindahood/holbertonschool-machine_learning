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
        """Create the VGG19-based model used to calculate cost"""

        # Load VGG19 without top layers
        vgg = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet'
        )

        # Freeze the model
        vgg.trainable = False

        # Collect outputs for style and content layers
        outputs = [vgg.get_layer(layer).output for layer in self.style_layers]
        outputs.append(vgg.get_layer(self.content_layer).output)

        # Create the model
        self.model = tf.keras.Model(inputs=vgg.input, outputs=outputs)
