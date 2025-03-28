#!/usr/bin/env python3
""" script 1"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a fully connected neural network using the Keras Functional API.
    """

    inputs = K.Input(shape=(nx,))

    reg = K.regularizers.L1L2(l2=lambtha)

    my_layer = K.layers.Dense(units=layers[0],
                              activation=activations[0],
                              kernel_regularizer=reg,
                              input_shape=(nx,))(inputs)

    for i in range(1, len(layers)):
        my_layer = K.layers.Dropout(1 - keep_prob)(my_layer)
        my_layer = K.layers.Dense(units=layers[i],
                                  activation=activations[i],
                                  kernel_regularizer=reg,
                                  )(my_layer)

    model = K.Model(inputs=inputs, outputs=my_layer)

    return model
