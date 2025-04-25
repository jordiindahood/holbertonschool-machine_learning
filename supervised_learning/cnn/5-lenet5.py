#!/usr/bin/env python3
""" script 5  """
from tensorflow import keras as K


def lenet5(X):
    """
    Creates a LeNet-5 model using Keras.
    """
    initializer = K.initializers.HeNormal(seed=0)

    model = K.Sequential([
        X,
        K.layers.Conv2D(filters=6,
                        kernel_size=5,
                        padding='same',
                        activation='relu',
                        kernel_initializer=initializer),
        K.layers.MaxPool2D(pool_size=2, strides=2),
        K.layers.Conv2D(filters=16,
                        kernel_size=5,
                        activation='relu',
                        kernel_initializer=initializer),
        K.layers.MaxPool2D(pool_size=2, strides=2),
        K.layers.Flatten(),
        K.layers.Dense(units=120,
                       activation='relu',
                       kernel_initializer=initializer),
        K.layers.Dense(units=84,
                       activation='relu',
                       kernel_initializer=initializer),
        K.layers.Dense(units=10,
                       activation='softmax',
                       kernel_initializer=initializer)
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.Adam(),
                  metrics=['accuracy'])

    return model
