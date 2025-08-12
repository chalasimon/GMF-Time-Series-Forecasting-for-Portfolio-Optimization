import os
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

DATA_DIR = "../data"

def load_asset_data(tickers):
    """
    Load CSV files for given tickers into a dict of DataFrames.
    """
    dfs = {}
    for ticker in tickers:
        file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        dfs[ticker] = df
    return dfs
