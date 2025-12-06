#!/usr/bin/env python3
""" script 13 """

import pandas as pd


def analyze(df):
    """
    Computes descriptive statistics for all columns except Timestamp.
    Returns a new DataFrame containing these statistics.
    """

    # Drop Timestamp if it exists
    if "Timestamp" in df.columns:
        df_numeric = df.drop(columns=["Timestamp"])
    else:
        df_numeric = df

    # Compute descriptive statistics
    stats = df_numeric.describe()

    return stats
