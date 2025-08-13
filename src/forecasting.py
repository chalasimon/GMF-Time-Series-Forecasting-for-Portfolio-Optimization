"""
Forecasting utilities for GMF Interim (Task 2)
- Chronological split
- ARIMA (auto_arima)
- LSTM (Keras)
- Metrics & plotting
"""
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

import matplotlib.pyplot as plt
import seaborn as sns


def chronological_split(series: pd.Series, split_date: str):
    """Split a price series into train/test by date string 'YYYY-MM-DD'."""
    series = series.sort_index()
    train = series.loc[:split_date]
    test = series.loc[split_date:]
    # remove the split date from train if duplicated boundary
    if not train.empty and not test.empty and train.index[-1] == test.index[0]:
        test = test.iloc[1:]
    return train, test