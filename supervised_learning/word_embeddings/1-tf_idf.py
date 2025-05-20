#!/usr/bin/env python3
"""script 1"""
from sklearn.feature_extraction.text import TfidfVectorizer
import re


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix using sklearn.
    """

    vectorizer = TfidfVectorizer(
        vocabulary=sorted(set(vocab)) if vocab else None
    )
    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names_out().tolist()

    return embeddings, features
