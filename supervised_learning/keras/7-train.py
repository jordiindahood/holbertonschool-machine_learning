#!/usr/bin/env python3
""" Task 7: 7. Learning Rate Decay """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent with optional
    early stopping and learning rate decay.
    """
    def inv_time_decay(epochs):
        """
        Updates the learning rate using inverse time decay.
        """
        return alpha / (1 + decay_rate * epochs)

    callbacks = []
    if (validation_data and learning_rate_decay):
        early_stopping = K.callbacks.LearningRateScheduler(inv_time_decay, 1)
        callbacks.append(early_stopping)

    if (validation_data and early_stopping):
        early_stopping = K.callbacks.EarlyStopping(patience=patience)
        callbacks.append(early_stopping)

    history = network.fit(data,
                          labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          verbose=verbose,
                          shuffle=shuffle,
                          callbacks=callbacks)

    return history
