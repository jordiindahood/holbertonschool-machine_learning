#!/usr/bin/env python3
""" script 3 """
import pandas as pd


def rename(df):
    """Renames Timestamp column to Datetime, converts it,
    and returns only Datetime and Close columns."""

    # Rename the column
    df = df.rename(columns={"Timestamp": "Datetime"})

    # Convert Unix timestamp to datetime
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit='s')

    # Return only the required columns
    return df[["Datetime", "Close"]]
