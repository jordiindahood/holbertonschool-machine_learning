#!/usr/bin/env python3
""" script 4  """
from tensorflow import keras as K

identity_block = __import__("2-identity_block").identity_block
projection_block = __import__("3-projection_block").projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as a Keras Model.
    """
    initializer = K.initializers.he_normal(seed=0)

    X = K.Input(shape=(224, 224, 3))

    layer = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding="same",
        kernel_initializer=initializer,
    )(X)

    layer = K.layers.BatchNormalization(axis=3)(layer)
    layer = K.layers.ReLU()(layer)

    layer = K.layers.MaxPool2D(pool_size=(3, 3),
                               padding="same", strides=(2, 2))(layer)

    layer = projection_block(layer, [64, 64, 256], 1)
    for i in range(2):
        layer = identity_block(layer, [64, 64, 256])

    layer = projection_block(layer, [128, 128, 512])
    for i in range(3):
        layer = identity_block(layer, [128, 128, 512])

    layer = projection_block(layer, [256, 256, 1024])
    for i in range(5):
        layer = identity_block(layer, [256, 256, 1024])

    layer = projection_block(layer, [512, 512, 2048])
    for i in range(2):
        layer = identity_block(layer, [512, 512, 2048])

    layer = K.layers.AveragePooling2D(pool_size=(7, 7),
                                      padding="same")(layer)

    layer = K.layers.Dense(
        units=1000,
        activation="softmax",
        kernel_initializer=initializer,
    )(layer)

    myModel = K.models.Model(inputs=X, outputs=layer)

    return myModel
