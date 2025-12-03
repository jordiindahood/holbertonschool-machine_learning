#!/usr/bin/env python3
""" script2 """
import pandas as pd


def from_file(filename, delimiter):
    """Loads data from a file as a pandas DataFrame.

    Args:
        filename (str): path to the file
        delimiter (str): column separator

    Returns:
        pd.DataFrame: loaded DataFrame
    """
    return pd.read_csv(filename, delimiter=delimiter)
