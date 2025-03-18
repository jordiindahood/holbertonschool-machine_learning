#!/usr/bin/env python3
""" Task 0: 0. Sequential """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a fully connected neural network using the Keras library.
    """
    model = K.Sequential()

    reg = K.regularizers.L1L2(l2=lambtha)

    model.add(K.layers.Dense(units=layers[0],
                             activation=activations[0],
                             kernel_regularizer=reg,
                             input_shape=(nx,),))

    for i in range(1, len(layers)):
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(K.layers.Dense(units=layers[i],
                                 activation=activations[i],
                                 kernel_regularizer=reg,
                                 ))

    return model
