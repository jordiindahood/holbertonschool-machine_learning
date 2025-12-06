#!/usr/bin/env python3

""" script 12 """

import pandas as pd

index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Build a Timestamp-first MultiIndex of bitstamp + coinbase
    from timestamps 1417411980 to 1417417980.
    """

    # Index both DataFrames on Timestamp
    df1 = index(df1)
    df2 = index(df2)

    # Define the timestamp range
    start = 1417411980
    end = 1417417980

    # Slice both dataframes for this range
    df1_slice = df1.loc[(df1.index >= start) & (df1.index <= end)]
    df2_slice = df2.loc[(df2.index >= start) & (df2.index <= end)]

    # Concatenate using keys (bitstamp first, then coinbase)
    combined = pd.concat(
        [df2_slice, df1_slice], keys=["bitstamp", "coinbase"], axis=0
    )

    # Swap levels so Timestamp becomes the first level
    combined = combined.swaplevel(0, 1)

    # Sort by Timestamp (outer index)
    combined = combined.sort_index(level=0)

    return combined
