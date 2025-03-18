#!/usr/bin/env python3
""" Task 0: 0. Sequential """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a fully connected neural network using the Keras library.
    """
    model = K.Sequential()

    model.add(
        K.layers.Dense(
            layers[0],
            activation=activations[0],
            kernel_regularizer=K.regularizers.l2(lambtha),
            input_shape=(nx,),
        )
    )

    model.add(K.layers.Dropout(1 - keep_prob))

    for i in range(1, len(layers)):
        model.add(
            K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha),
            )
        )


    return model
