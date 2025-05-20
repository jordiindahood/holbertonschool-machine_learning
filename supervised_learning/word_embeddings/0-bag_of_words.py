#!/usr/bin/env python3
"""script 0"""

import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag-of-words embedding matrix
    """

    def tokenize(sentence):
        return re.findall(r'\b\w+\b', sentence.lower())

    tokenized = [tokenize(sent) for sent in sentences]

    if vocab is None:
        unique_words = set()
        for tokens in tokenized:
            unique_words.update(tokens)
        features = sorted(unique_words)
    else:
        features = sorted(set(vocab))

    word_to_index = {word: i for i, word in enumerate(features)}

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    for i, tokens in enumerate(tokenized):
        for word in tokens:
            if word in word_to_index:
                embeddings[i, word_to_index[word]] += 1

    return embeddings, features
