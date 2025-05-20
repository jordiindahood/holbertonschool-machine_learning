#!/usr/bin/env python3
"""script 1"""
from sklearn.feature_extraction.text import TfidfVectorizer
import re


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix using sklearn.
    """

    def preprocess(sentence):
        return ' '.join(re.findall(r'\b\w+\b', sentence.lower()))

    cleaned_sentences = [preprocess(s) for s in sentences]

    vectorizer = TfidfVectorizer(
        vocabulary=sorted(set(vocab)) if vocab else None
    )
    embeddings = vectorizer.fit_transform(cleaned_sentences).toarray()
    features = vectorizer.get_feature_names_out().tolist()

    return embeddings, features
