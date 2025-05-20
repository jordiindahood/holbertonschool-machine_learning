#!/usr/bin/env python3
"""script 3"""

import tensorflow as tf


def gensim_to_keras(model):
    """
    converts a gensim word2vec model to a
    keras Embedding layer
    """

    weights = tf.convert_to_tensor(
        [model.wv[word] for word in model.wv.index_to_key]
    )
    vocab_size, vector_size = weights.shape

    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        weights=[weights.numpy()],
        trainable=True,
    )
    return embedding_layer
