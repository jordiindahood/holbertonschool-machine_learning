#!/usr/bin/env python3
""" script 6 """


def flip_switch(df):
    """Sorts the DataFrame in reverse chronological order
    (by Timestamp descending) and transposes it."""

    # Sort reverse chronological (largest timestamp first)
    df = df.sort_values(by="Timestamp", ascending=False)

    # Transpose the sorted DataFrame
    return df.T
