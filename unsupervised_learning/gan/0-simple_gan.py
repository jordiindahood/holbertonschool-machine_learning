#!/usr/bin/env python3
"""
Simple GAN implementation using TensorFlow / Keras.

This module defines a minimal Generative Adversarial Network (GAN)
implemented by subclassing `keras.Model`. The GAN is composed of:
- a generator network
- a discriminator network
- a custom training loop implemented via `train_step`

The GAN is trained using Mean Squared Error (MSE) losses and Adam
optimizers for both generator and discriminator.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class Simple_GAN(keras.Model):
    """
    Simple Generative Adversarial Network (GAN).

    This class implements a GAN by combining:
    - a generator model
    - a discriminator model
    - a latent space generator
    - a fixed dataset of real examples

    The training alternates between:
    - multiple discriminator updates
    - one generator update

    The class overrides `train_step` to define custom GAN training logic.
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
    ):
        """
        Initialize the Simple_GAN model.

        Parameters
        ----------
        generator : keras.Model
            Neural network that maps latent vectors to fake samples.

        discriminator : keras.Model
            Neural network that scores real and fake samples.

        latent_generator : callable
            Function that generates latent vectors of shape
            (batch_size, latent_dim).

        real_examples : tf.Tensor or np.ndarray
            Dataset containing real samples used for training.

        batch_size : int, optional
            Number of samples per training batch (default: 200).

        disc_iter : int, optional
            Number of discriminator updates per generator update
            (default: 2).

        learning_rate : float, optional
            Learning rate for both generator and discriminator optimizers
            (default: 0.005).
        """
        super().__init__()  # Initialize keras.Model

        # Store core components
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator

        # Training hyperparameters
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate

        # Adam optimizer parameters (standard GAN values)
        self.beta_1 = 0.5
        self.beta_2 = 0.9

        # -------------------------
        # Generator configuration
        # -------------------------
        # Generator wants discriminator outputs close to +1
        self.generator.loss = lambda x: tf.keras.losses.MeanSquaredError()(
            x, tf.ones(x.shape)
        )

        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )

        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss,
        )

        # -----------------------------
        # Discriminator configuration
        # -----------------------------
        # Discriminator wants:
        #  - real samples → +1
        #  - fake samples → -1
        self.discriminator.loss = (
            lambda x, y: tf.keras.losses.MeanSquaredError()(
                x, tf.ones(x.shape)
            )
            + tf.keras.losses.MeanSquaredError()(y, -1 * tf.ones(y.shape))
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
        Generate fake samples using the generator.

        Parameters
        ----------
        size : int or None
            Number of fake samples to generate.
            Defaults to `self.batch_size`.

        training : bool
            Whether the generator should behave in training mode.

        Returns
        -------
        tf.Tensor
            Generated fake samples.
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """
        Sample real examples uniformly at random from the dataset.

        Parameters
        ----------
        size : int or None
            Number of real samples to return.
            Defaults to `self.batch_size`.

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

    def train_step(self, useless_argument):
        """
        Perform one GAN training step.

        This method overrides `keras.Model.train_step` and is called
        automatically by `model.fit()`.

        Training procedure:
        1. Update the discriminator `disc_iter` times
        2. Update the generator once

        Parameters
        ----------
        useless_argument : any
            Required by Keras API but not used.
            (Data is sampled internally.)

        Returns
        -------
        dict
            Dictionary containing discriminator and generator losses.
        """

        # ===============================
        # Train the discriminator
        # ===============================
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                # Sample real and fake data
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)

                # Discriminator predictions
                real_out = self.discriminator(real_samples, training=True)
                fake_out = self.discriminator(fake_samples, training=True)

                # Compute discriminator loss
                discr_loss = self.discriminator.loss(real_out, fake_out)

            # Apply discriminator gradients
            grads = tape.gradient(
                discr_loss, self.discriminator.trainable_variables
            )
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables)
            )

        # ===============================
        # Train the generator
        # ===============================
        with tf.GradientTape() as tape:
            # Generate fake samples
            fake_samples = self.get_fake_sample(training=True)

            # Discriminator evaluates fake samples
            fake_out = self.discriminator(fake_samples, training=True)

            # Generator loss
            gen_loss = self.generator.loss(fake_out)

        # Apply generator gradients
        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

        return {
            "discr_loss": discr_loss,
            "gen_loss": gen_loss,
        }
