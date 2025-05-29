#!/usr/bin/env python3
"""script 3"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a trained Gensim Word2Vec model to a Keras Embedding layer.
    """
    weights = model.wv.vectors

    vocab_size, embedding_dim = weights.shape

    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[weights],
        trainable=True,
    )

    return embedding_layer
