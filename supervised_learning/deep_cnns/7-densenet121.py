#!/usr/bin/env python3
""" script 7 """
from tensorflow import keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture.
    """
    he_init = K.initializers.VarianceScaling(scale=2.0, seed=0)

    inputs = K.Input(shape=(224, 224, 3))

    # Initial Convolution and MaxPooling
    bn = K.layers.BatchNormalization()(inputs)
    act = K.layers.Activation('relu')(bn)
    conv = K.layers.Conv2D(
        64, kernel_size=7, strides=2, padding='same',
        kernel_initializer=he_init
    )(act)
    pool = K.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(conv)

    # Dense blocks and transition layers
    num_blocks = [6, 12, 24, 16]  # number of layers in each dense block

    x, nb_filters = dense_block(pool, 64, growth_rate, num_blocks[0])
    x, nb_filters = transition_layer(x, nb_filters, compression)

    x, nb_filters = dense_block(x, nb_filters, growth_rate, num_blocks[1])
    x, nb_filters = transition_layer(x, nb_filters, compression)

    x, nb_filters = dense_block(x, nb_filters, growth_rate, num_blocks[2])
    x, nb_filters = transition_layer(x, nb_filters, compression)

    x, nb_filters = dense_block(x, nb_filters, growth_rate, num_blocks[3])

    # Final Batch Norm + ReLU + Global Average Pooling
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.GlobalAveragePooling2D()(x)

    # Fully connected classification layer
    outputs = K.layers.Dense(1000, activation='softmax', kernel_initializer=he_init)(x)

    model = K.Model(inputs=inputs, outputs=outputs)
    return model
