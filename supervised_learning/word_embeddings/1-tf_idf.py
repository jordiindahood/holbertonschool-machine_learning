#!/usr/bin/env python3
"""script 1"""

import numpy as np
import re
import math


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix
    """

    def tokenize(sentence):
        return re.findall(r'\b\w+\b', sentence.lower())

    tokenized = [tokenize(s) for s in sentences]
    s = len(sentences)

    if vocab is None:
        unique_words = set()
        for tokens in tokenized:
            unique_words.update(tokens)
        features = sorted(unique_words)
    else:
        features = sorted(set(vocab))

    f = len(features)
    word_to_index = {word: i for i, word in enumerate(features)}

    tf = np.zeros((s, f))
    for i, tokens in enumerate(tokenized):
        total = len(tokens)
        for word in tokens:
            if word in word_to_index:
                tf[i, word_to_index[word]] += 1
        if total > 0:
            tf[i] /= total

    idf = np.zeros(f)
    for j, word in enumerate(features):
        doc_count = sum(1 for doc in tokenized if word in doc)
        if doc_count > 0:
            idf[j] = math.log(s / doc_count)
        else:
            idf[j] = 0.0

    embeddings = tf * idf

    return embeddings, features
