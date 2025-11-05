#!/usr/bin/env python3
"""Convert a trained gensim Word2Vec model to a Keras Embedding layer"""

import numpy as np
import tensorflow


def gensim_to_keras(model):
    """
    Converts a gensim Word2Vec model to a Keras Embedding layer.
    """
    weights = model.wv.vectors
    vocab_size, vector_size = weights.shape

    embedding_layer = tensorflow.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        weights=[weights],
        trainable=True,
    )

    return embedding_layer
