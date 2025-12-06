#!/usr/bin/env python3
""" script 7 """


def high(df):
    """Sorts a DataFrame by the High column in descending order."""

    return df.sort_values(by="High", ascending=False)
