#!/usr/bin/env python3
""" script 5 """


def slice(df):
    """Extracts columns and selects every 60th row."""

    # Select required columns
    cols = df[["High", "Low", "Close", "Volume_(BTC)"]]

    # Select every 60th row
    return cols.iloc[::60]
