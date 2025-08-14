# import necessary libraries
import numpy as np
import pandas as pd
from dataclasses import dataclass
import matplotlib.pyplot as plt

# PyPortfolioOpt for stable optimization
try:
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt import risk_models, expected_returns
    from pypfopt import objective_functions
    HAS_PFOPT = True
except Exception:
    HAS_PFOPT = False


# ------------------------------
# Utilities
# ------------------------------

def compute_daily_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily percentage returns from price dataframe (Adj Close preferred)."""
    # Assumes columns are ['TSLA','BND','SPY'] price series
    ret = price_df.sort_index().pct_change().dropna(how="all")
    return ret


def annualize_mean_return(daily_returns: pd.Series) -> float:
    """Annualize the mean daily return: mean * 252."""
    return daily_returns.mean() * 252.0


def annualize_covariance(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Annualize the covariance matrix: cov * 252."""
    return daily_returns.cov() * 252.0


def expected_return_from_tsla_forecast(tsla_forecast: pd.DataFrame) -> float:
    """
    Turn a TSLA future price forecast into an expected annual return.
    Expects a DataFrame with a 'mean' column over future dates.
    Strategy:
      1) Convert future prices to daily pct_change returns
      2) Take the average daily return
      3) Annualize (×252)
    """
    # Accept flexible column names
    if "mean" in tsla_forecast.columns:
        price = tsla_forecast["mean"].copy()
    elif "ARIMA_Forecast" in tsla_forecast.columns:
        price = tsla_forecast["ARIMA_Forecast"].copy()
    elif "LSTM_Forecast" in tsla_forecast.columns:
        price = tsla_forecast["LSTM_Forecast"].copy()
    else:
        # If a Series is passed or a DF without those names, try first column
        if isinstance(tsla_forecast, pd.Series):
            price = tsla_forecast.copy()
        else:
            price = tsla_forecast.iloc[:, 0].copy()

    future_rets = price.pct_change().dropna()
    if future_rets.empty:
        raise ValueError("TSLA forecast did not produce non-empty returns. Check inputs.")
    return annualize_mean_return(future_rets)