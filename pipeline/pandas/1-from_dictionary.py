#!/usr/bin/env python3
""" script 1 """
import pandas as pd

# Create the DataFrame from a dictionary
df = pd.DataFrame(
    {
        "First": [0.0, 0.5, 1.0, 1.5],
        "Second": ["one", "two", "three", "four"],
    },
    index=["A", "B", "C", "D"],
)
