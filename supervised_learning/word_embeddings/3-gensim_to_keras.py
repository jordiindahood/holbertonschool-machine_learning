#!/usr/bin/env python3
"""script 3"""

import numpy as np
from tensorflow.keras.layers import Embedding


def gensim_to_keras(model):
    """converts a gensim word2vec model to a
    keras Embedding layer"""
    weights = model.wv.vectors

    vocab_size, vector_size = weights.shape

    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        weights=[weights],
        trainable=True,
    )

    return embedding_layer
