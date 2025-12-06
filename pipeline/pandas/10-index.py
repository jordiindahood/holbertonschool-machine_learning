#!/usr/bin/env python3
""" sxript 10 """


def index(df):
    """
    Sets the Timestamp column as the index of the DataFrame.
    Returns the modified DataFrame.
    """
    df = df.set_index("Timestamp")
    return df
