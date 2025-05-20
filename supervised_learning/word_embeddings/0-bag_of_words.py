#!/usr/bin/env python3
"""script 0"""

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag-of-words embedding matrix
    """

    vectorizer = CountVectorizer(vocabulary=vocab)
    x = vectorizer.fit_transform(sentences)
    embedding = x.toarray()
    features = vectorizer.get_feature_names_out()

    return embedding, features
