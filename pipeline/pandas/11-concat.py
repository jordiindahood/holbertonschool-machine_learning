#!/usr/bin/env python3
""" script 11 """
index = __import__('10-index').index
import pandas as pd


def concat(df1, df2):
    """
    Concatenate bitstamp data (df2) up to timestamp 1417411920
    above the coinbase data (df1), using Timestamp as index.

    Bitstamp rows should be labeled 'bitstamp'
    Coinbase rows should be labeled 'coinbase'
    """
    # Index both dataframes on Timestamp
    df1 = index(df1)
    df2 = index(df2)

    # Select only bitstamp rows where timestamp <= 1417411920
    df2_cut = df2.loc[df2.index <= 1417411920]

    # Concatenate with keys
    result = pd.concat([df2_cut, df1], axis=0, keys=["bitstamp", "coinbase"])

    return result
