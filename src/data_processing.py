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
def clean_data(dfs):
    """
    For each asset DataFrame:
    - check datatypes
    - fill missing values with forward fill then backward fill
    """
    for ticker, df in dfs.items():
        # Ensure numeric types
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Fill missing values
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
    return dfs
def calculate_returns(dfs):
    """
    Calculate daily percentage returns (simple returns) for each asset.
    Returns a DataFrame with columns = tickers.
    """
    returns = pd.DataFrame()
    for ticker, df in dfs.items():
        returns[ticker] = df['Close'].pct_change()
    return returns.dropna()
def rolling_volatility(returns, window=21):
    """
    Calculate rolling annualized volatility (std dev) of returns.
    window = number of trading days (default 21 = ~1 month).
    Annualized by sqrt(252).
    """
    vol = returns.rolling(window=window).std() * np.sqrt(252)
    return vol
