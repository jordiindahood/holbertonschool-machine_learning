#!/usr/bin/env python3
"""
Wasserstein GAN with weight clipping (WGAN-clip).

This module defines a Wasserstein GAN implemented by subclassing
`keras.Model`. The discriminator acts as a critic and its weights
are clipped in a fixed range to enforce the 1-Lipschitz constraint.

The training procedure alternates between multiple critic updates
and one generator update, following the original WGAN paper.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_clip(keras.Model):
    """
    Wasserstein Generative Adversarial Network with weight clipping.

    This class implements a WGAN where:
    - the discriminator is a critic (no sigmoid output)
    - the loss functions follow the Wasserstein formulation
    - the critic's weights are clipped in the range [-1, 1]

    The model overrides `train_step` to define a custom training loop.
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
        Initialize the WGAN with weight clipping.

        Parameters
        ----------
        generator : keras.Model
            Generator network mapping latent vectors to fake samples.

        discriminator : keras.Model
            Critic network scoring real and fake samples.

        latent_generator : callable
            Function generating latent vectors of shape
            (batch_size, latent_dim).

        real_examples : tf.Tensor or np.ndarray
            Dataset of real samples used during training.

        batch_size : int, optional
            Number of samples per training batch (default: 200).

        disc_iter : int, optional
            Number of critic updates per generator update (default: 2).

        learning_rate : float, optional
            Learning rate for both optimizers (default: 0.005).
        """
        super().__init__()  # Initialize keras.Model

        # Core components
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator

        # Training parameters
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate

        # Adam optimizer parameters (standard WGAN values)
        self.beta_1 = 0.5
        self.beta_2 = 0.9

        # -------------------------
        # Generator configuration
        # -------------------------
        # WGAN generator loss: -E[D(fake)]
        self.generator.loss = lambda x: -tf.math.reduce_mean(x)

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
        # Critic (discriminator) configuration
        # -----------------------------
        # WGAN critic loss: E[D(fake)] - E[D(real)]
        self.discriminator.loss = lambda x, y: tf.math.reduce_mean(
            x
        ) - tf.math.reduce_mean(y)

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
            Whether to run the generator in training mode.

        Returns
        -------
        tf.Tensor
            Batch of generated fake samples.
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
        Perform one training step of the WGAN.

        This method overrides `keras.Model.train_step` and is invoked
        automatically by `model.fit()`.

        Training procedure:
        1. Update the critic `disc_iter` times
        2. Clip the critic weights after each update
        3. Update the generator once

        Parameters
        ----------
        useless_argument : any
            Required by Keras API but not used.
            (Data is sampled internally.)

        Returns
        -------
        dict
            Dictionary containing critic and generator losses.
        """

        # -------------------------------
        # Train the critic (discriminator)
        # -------------------------------
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)

                real_out = self.discriminator(real_samples, training=True)
                fake_out = self.discriminator(fake_samples, training=True)

                # Critic loss
                discr_loss = self.discriminator.loss(fake_out, real_out)

            # Apply critic gradients
            grads = tape.gradient(
                discr_loss, self.discriminator.trainable_variables
            )
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables)
            )

            # Enforce 1-Lipschitz constraint via weight clipping
            for var in self.discriminator.trainable_variables:
                var.assign(tf.clip_by_value(var, -1.0, 1.0))

        # -------------------------------
        # Train the generator
        # -------------------------------
        with tf.GradientTape() as tape:
            fake_samples = self.get_fake_sample(training=True)
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
