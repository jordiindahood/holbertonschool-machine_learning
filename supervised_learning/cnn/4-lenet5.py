#!/usr/bin/env python3
""" script 4 """
import tensorflow.compat.v1 as tf # type: ignore


def lenet5(x, y):
    """
    Builds the LeNet-5 architecture using TensorFlow for digit classification using TF 1.

    """
    # He initialization
    he_init = tf.keras.initializers.VarianceScaling(scale=2.0)

    # Layer 1: Conv -> ReLU
    conv1 = tf.compat.v1.layers.conv2d(
        inputs=x, filters=6, kernel_size=5, padding='same',
        activation=tf.nn.relu, kernel_initializer=he_init
    )

    # Layer 2: Max Pooling
    pool1 = tf.compat.v1.layers.max_pooling2d(
        inputs=conv1, pool_size=2, strides=2
    )

    # Layer 3: Conv -> ReLU
    conv2 = tf.compat.v1.layers.conv2d(
        inputs=pool1, filters=16, kernel_size=5, padding='valid',
        activation=tf.nn.relu, kernel_initializer=he_init
    )

    # Layer 4: Max Pooling
    pool2 = tf.compat.v1.layers.max_pooling2d(
        inputs=conv2, pool_size=2, strides=2
    )

    # Flatten
    flat = tf.compat.v1.layers.flatten(pool2)

    # Layer 5: Fully Connected -> ReLU
    fc1 = tf.compat.v1.layers.dense(
        inputs=flat, units=120,
        activation=tf.nn.relu, kernel_initializer=he_init
    )

    # Layer 6: Fully Connected -> ReLU
    fc2 = tf.compat.v1.layers.dense(
        inputs=fc1, units=84,
        activation=tf.nn.relu, kernel_initializer=he_init
    )

    # Output Layer: Fully Connected -> Softmax
    logits = tf.compat.v1.layers.dense(
        inputs=fc2, units=10,
        kernel_initializer=he_init
    )

    # Softmax output
    y_pred = tf.nn.softmax(logits)

    # Loss: softmax cross-entropy
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

    # Optimizer: Adam
    train_op = tf.compat.v1.train.AdamOptimizer().minimize(loss)

    # Accuracy
    correct_preds = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

    return y_pred, train_op, loss, accuracy
