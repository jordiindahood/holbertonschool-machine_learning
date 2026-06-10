#!/usr/bin/env python3
"""
WGAN-GP implementation using TensorFlow / Keras.

This module defines a WGAN_GP class that implements a Wasserstein GAN
with Gradient Penalty (WGAN-GP) by subclassing keras.Model and
overriding the train_step method.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_GP(keras.Model):
    """
    Wasserstein GAN with Gradient Penalty (WGAN-GP).

    This class implements a WGAN-GP model, where:
    - The discriminator (critic) estimates Wasserstein distance
    - The Lipschitz constraint is enforced using a gradient penalty
    - Training is performed via a custom train_step method

    The class is compatible with keras.Model.fit().
    """

    def __init__(
        self,
        generator,
        discriminator,
        latent_generator,
        real_examples,
        batch_size=200,
        disc_iter=2,
        learning_rate=0.005,
        lambda_gp=10,
    ):
        """
        Initialize the WGAN-GP model.

        Parameters
        ----------
        generator : keras.Model
            Generator network mapping latent vectors to fake samples.
        discriminator : keras.Model
            Critic network mapping samples to scalar scores.
        latent_generator : callable
            Function that generates latent vectors given a batch size.
        real_examples : tf.Tensor
            Tensor containing the real dataset samples.
        batch_size : int, optional
            Number of samples per batch (default is 200).
        disc_iter : int, optional
            Number of discriminator updates per generator update.
        learning_rate : float, optional
            Learning rate for both optimizers.
        lambda_gp : float, optional
            Weight of the gradient penalty term.
        """
        super().__init__()

        # Core components
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator

        # Training parameters
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate

        # Adam optimizer parameters (recommended for WGAN-GP)
        self.beta_1 = 0.3
        self.beta_2 = 0.9

        # Gradient penalty parameters
        self.lambda_gp = lambda_gp

        # Shape utilities for gradient penalty computation
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, dtype="int32")

        # Shape used to sample interpolation coefficients
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

        # --------------------------------------------------
        # Generator loss and optimizer
        # --------------------------------------------------
        self.generator.loss = lambda x: -tf.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )
        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss,
        )

        # --------------------------------------------------
        # Discriminator (critic) loss and optimizer
        # --------------------------------------------------
        self.discriminator.loss = lambda x, y: (
            tf.reduce_mean(x) - tf.reduce_mean(y)
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )
        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss,
        )

    def get_fake_sample(self, size=None, training=False):
        """
        Generate a batch of fake samples using the generator.

        Parameters
        ----------
        size : int, optional
            Number of samples to generate (defaults to batch_size).
        training : bool, optional
            Whether the generator is in training mode.

        Returns
        -------
        tf.Tensor
            Generated fake samples.
        """
        if not size:
            size = self.batch_size
        return self.generator(
            self.latent_generator(size),
            training=training,
        )

    def get_real_sample(self, size=None):
        """
        Sample a batch of real examples from the dataset.

        Parameters
        ----------
        size : int, optional
            Number of samples to draw (defaults to batch_size).

        Returns
        -------
        tf.Tensor
            Batch of real samples.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def get_interpolated_sample(self, real_sample, fake_sample):
        """
        Generate interpolated samples between real and fake samples.

        This is used for computing the gradient penalty.

        Parameters
        ----------
        real_sample : tf.Tensor
            Batch of real samples.
        fake_sample : tf.Tensor
            Batch of fake samples.

        Returns
        -------
        tf.Tensor
            Interpolated samples.
        """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """
        Compute the gradient penalty for WGAN-GP.

        The penalty enforces the 1-Lipschitz constraint by penalizing
        deviations of the gradient norm from 1.

        Parameters
        ----------
        interpolated_sample : tf.Tensor
            Interpolated samples between real and fake data.

        Returns
        -------
        tf.Tensor
            Scalar gradient penalty value.
        """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)

        grads = gp_tape.gradient(pred, interpolated_sample)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, useless_argument):
        """
        Perform one training step of WGAN-GP.

        This method:
        - Updates the discriminator (critic) disc_iter times
        - Applies the gradient penalty
        - Updates the generator once

        Parameters
        ----------
        useless_argument : Any
            Ignored. Required by Keras API.

        Returns
        -------
        dict
            Dictionary containing discriminator loss, generator loss,
            and gradient penalty.
        """

        # -------------------------------
        # Train the discriminator (critic)
        # -------------------------------
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)

                real_out = self.discriminator(real_samples, training=True)
                fake_out = self.discriminator(fake_samples, training=True)

                discr_loss = self.discriminator.loss(fake_out, real_out)

                interpolated = self.get_interpolated_sample(
                    real_samples, fake_samples
                )
                gp = self.gradient_penalty(interpolated)

                new_discr_loss = discr_loss + self.lambda_gp * gp

            grads = tape.gradient(
                new_discr_loss,
                self.discriminator.trainable_variables,
            )
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables)
            )

        # -------------------------------
        # Train the generator
        # -------------------------------
        with tf.GradientTape() as tape:
            fake_samples = self.get_fake_sample(training=True)
            fake_out = self.discriminator(fake_samples, training=True)
            gen_loss = self.generator.loss(fake_out)

        grads = tape.gradient(
            gen_loss,
            self.generator.trainable_variables,
        )
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

        return {
            "discr_loss": discr_loss,
            "gen_loss": gen_loss,
            "gp": gp,
        }
