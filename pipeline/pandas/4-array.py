#!/usr/bin/env python3
""" script 4 """

def array(df):
    """Selects the last 10 rows of High and Close columns
    and returns them as a numpy ndarray."""

    # Select last 10 rows from High and Close
    sub = df[["High", "Close"]].tail(10)

    # Convert to numpy array
    return sub.to_numpy()
