#!/usr/bin/env python3
""" script 8 """


def prune(df):
    """
    Removes entries where Close has NaN values.
    Returns the modified DataFrame.
    """
    return df.dropna(subset=['Close'])
