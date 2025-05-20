#!/usr/bin/env python3
"""script 3"""

import tensorflow as tf


def gensim_to_keras(model):
    """
    converts a gensim word2vec model to a
    keras Embedding layer
    """

    weights = model.wv.vectors

    vocab_size, embedding_dim = weights.shape

    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[weights],
        trainable=True
    )

    return embedding_layer