#!/usr/bin/env python3
""" script 3 """
shuffle_data = __import__("2-shuffle_data").shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches to be used for training
    """
    x_shuffled, y_shuffled = shuffle_data(X, Y)
    m = X.shape[0]
    mini_batches = list()
    if batch_size > m:
        batch_size = m
    for i in range(0, m, batch_size):
        X_batch = x_shuffled[i : i + batch_size]
        Y_batch = y_shuffled[i : i + batch_size]
        mini_batches.append((X_batch, Y_batch))
    return mini_batches
