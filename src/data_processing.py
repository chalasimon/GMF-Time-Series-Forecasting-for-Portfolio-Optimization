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
def normalize(df):
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    
    return df
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

def detect_outliers(returns, z_thresh=3):
    """
    Detect outliers based on Z-score of returns.
    Returns DataFrame of booleans where True indicates outlier days.
    """
    z_scores = (returns - returns.mean()) / returns.std()
    return (np.abs(z_scores) > z_thresh)
def get_extreme_return_days(returns, top_n=5):
    """
    Find top_n days with highest positive and negative returns.
    Returns two DataFrames: top positive and top negative return days.
    """
    # Flatten to long format: date, asset, return
    returns_long = returns.stack().reset_index()
    returns_long.columns = ['Date', 'Asset', 'Return']

    # Top positive returns
    top_pos = returns_long.nlargest(top_n, 'Return')

    # Top negative returns
    top_neg = returns_long.nsmallest(top_n, 'Return')

    return top_pos, top_neg
def adf_test(series, signif=0.05):
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    Returns dictionary with ADF statistic, p-value, and stationarity bool.
    """
    result = adfuller(series.dropna(), autolag='AIC')
    adf_stat, p_value = result[0], result[1]
    return {
        'adf_statistic': adf_stat,
        'p_value': p_value,
        'stationary': p_value < signif
    }
def sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calculate annualized Sharpe Ratio given daily returns.
    Assumes 252 trading days per year.
    """
    excess_returns = returns - risk_free_rate / 252
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

def value_at_risk(returns, confidence_level=0.05):
    """
    Calculate historical Value at Risk (VaR) at given confidence level.
    """
    return returns.quantile(confidence_level)