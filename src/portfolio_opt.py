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

# ------------------------------
# Optimization Outputs
# ------------------------------

@dataclass
class PortfolioPoint:
    risk: float
    ret: float
    sharpe: float
    weights: dict


@dataclass
class FrontierResults:
    points: list            # list[PortfolioPoint]
    max_sharpe: PortfolioPoint
    min_vol: PortfolioPoint


# ------------------------------
# Optimization (with/without PyPortfolioOpt)
# ------------------------------

def _opt_with_pypfopt(mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0) -> FrontierResults:
    assets = list(mu.index)

    # Max Sharpe
    ef1 = EfficientFrontier(mu, cov)
    ef1.add_objective(objective_functions.L2_reg, gamma=0.001)  # tiny regularization
    w_ms = ef1.max_sharpe(risk_free_rate=rf)
    w_ms = ef1.clean_weights()
    ret_ms, risk_ms, sharpe_ms = ef1.portfolio_performance(risk_free_rate=rf)
    max_sharpe_pt = PortfolioPoint(risk=risk_ms, ret=ret_ms, sharpe=sharpe_ms, weights=w_ms)

    # Min volatility
    ef2 = EfficientFrontier(mu, cov)
    ef2.add_objective(objective_functions.L2_reg, gamma=0.001)
    w_mv = ef2.min_volatility()
    w_mv = ef2.clean_weights()
    ret_mv, risk_mv, sharpe_mv = ef2.portfolio_performance(risk_free_rate=rf)
    min_vol_pt = PortfolioPoint(risk=risk_mv, ret=ret_mv, sharpe=sharpe_mv, weights=w_mv)

    # Efficient frontier curve by sweeping target returns
    min_ret = min(ret_mv, ret_ms)
    max_ret = max(ret_mv, ret_ms)
    targets = np.linspace(min_ret, max_ret, 50)

    points = []
    for tr in targets:
        try:
            ef = EfficientFrontier(mu, cov)
            ef.add_objective(objective_functions.L2_reg, gamma=0.001)
            ef.efficient_return(target_return=tr)
            R, S, Sh = ef.portfolio_performance(risk_free_rate=rf)
            w = ef.clean_weights()
            points.append(PortfolioPoint(risk=S, ret=R, sharpe=Sh, weights=w))
        except Exception:
            # Some targets are infeasible
            continue

    return FrontierResults(points=points, max_sharpe=max_sharpe_pt, min_vol=min_vol_pt)

