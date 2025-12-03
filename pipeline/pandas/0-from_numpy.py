#!/usr/bin/env python3
import pandas as pd


def from_numpy(array):
    """Creates a pd.DataFrame from a np.ndarray.

    Args:
        array (np.ndarray): input array

    Returns:
        pd.DataFrame: DataFrame with alphabetical column labels
    """
    # Number of columns in the array
    n_cols = array.shape[1]

    # Generate column names: "A", "B", ..., up to 26 max
    columns = [chr(ord('A') + i) for i in range(n_cols)]

    # Create and return DataFrame
    return pd.DataFrame(array, columns=columns)
