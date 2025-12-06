#!/usr/bin/env python3
""" script 9 """


def fill(df):
    """
    Cleans and fills the DataFrame:
    - Removes Weighted_Price column.
    - Fills missing Close with previous row's value.
    - Fills missing Open, High, Low with the row's Close.
    - Sets missing Volume_(BTC) and Volume_(Currency) to 0.
    Returns the modified DataFrame.
    """

    # 1. Remove Weighted_Price
    if "Weighted_Price" in df.columns:
        df = df.drop(columns=["Weighted_Price"])

    # 2. Fill Close missing values with previous row's value
    df["Close"] = df["Close"].fillna(method="ffill")

    # 3. Fill Open, High, Low missing values with the row's Close
    for col in ["Open", "High", "Low"]:
        df[col] = df[col].fillna(df["Close"])

    # 4. Replace missing volumes with 0
    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
    df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

    return df
