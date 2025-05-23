#!/usr/bin/env python3
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder."""

    # === ENCODER ===
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)

    # Mean and log variance layers
    z_mean = keras.layers.Dense(latent_dims)(x)
    z_log_var = keras.layers.Dense(latent_dims)(x)

    # Reparameterization trick
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = keras.backend.random_normal(
            shape=(keras.backend.shape(z_mean)[0], latent_dims),
            mean=0.0,
            stddev=1.0,
        )
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,))(
        [z_mean, z_log_var]
    )

    encoder = keras.Model(inputs, [z, z_mean, z_log_var], name='encoder')

    # === DECODER ===
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # === VAE MODEL ===
    outputs = decoder(z)
    vae = keras.Model(inputs, outputs, name='autoencoder')

    # Custom VAE loss
    reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss = keras.backend.sum(reconstruction_loss, axis=1)

    kl_loss = -0.5 * keras.backend.sum(
        1
        + z_log_var
        - keras.backend.square(z_mean)
        - keras.backend.exp(z_log_var),
        axis=1,
    )

    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    return encoder, decoder, vae
