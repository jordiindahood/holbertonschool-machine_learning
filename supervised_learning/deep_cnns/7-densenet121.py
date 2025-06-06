#!/usr/bin/env python3
""" script 7 """
from tensorflow import keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture.
    """

    initializer = K.initializers.HeNormal(seed=0)

    X = K.Input(shape=(224, 224, 3))
    layers = [12, 24, 16]

    myLayer = K.layers.BatchNormalization(axis=3)(X)
    myLayer = K.layers.ReLU()(myLayer)

    myLayer = K.layers.Conv2D(filters=2 * growth_rate,
                               kernel_size=(7, 7),
                               strides=(2, 2),
                               padding='same',
                               kernel_initializer=initializer)(myLayer)

    myLayer = K.layers.MaxPooling2D(pool_size=(3, 3),
                                     padding='same',
                                     strides=(2, 2))(myLayer)

    nb_filters = 2 * growth_rate

    myLayer, nb_filters = dense_block(myLayer, nb_filters, growth_rate, 6)

    for layer in layers:
        myLayer, nb_filters = transition_layer(myLayer,
                                                nb_filters,
                                                compression)

        myLayer, nb_filters = dense_block(myLayer,
                                           nb_filters,
                                           growth_rate,
                                           layer)

    myLayer = K.layers.AveragePooling2D(pool_size=(7, 7))(myLayer)

    myLayer = K.layers.Dense(units=1000,
                              activation='softmax',
                              kernel_initializer=initializer)(myLayer)

    model = K.models.Model(inputs=X, outputs=myLayer)

    return model
