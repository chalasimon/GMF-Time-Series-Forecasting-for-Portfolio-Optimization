"""
Task 3 – Future Forecasting Utilities (ARIMA + LSTM)
- Load saved models
- Produce 6–12 month (business days) forecasts
- Confidence intervals:
    * ARIMA: native conf_int from statsmodels
    * LSTM: Monte Carlo (MC) dropout or residual-based CI
- Plot helpers
"""

import os
import numpy as np
import pandas as pd
import pickle
from typing import Tuple

import matplotlib.pyplot as plt

# ARIMA (statsmodels)
from statsmodels.tsa.arima.model import ARIMAResults

# LSTM
from tensorflow.keras.models import load_model
from joblib import load as joblib_load


# -------------------------------
# Helpers
# -------------------------------

def _future_bdays(last_date: pd.Timestamp, steps: int) -> pd.DatetimeIndex:
    """Generate a business-day date index starting from the next business day."""
    start = pd.tseries.offsets.BusinessDay().apply(last_date)
    return pd.bdate_range(start=start, periods=steps)


# -------------------------------
# ARIMA – Load & Forecast
# -------------------------------

def load_arima(model_path: str):
    """Load a pickled statsmodels ARIMAResults."""
    with open(model_path, "rb") as f:
        model_fit: ARIMAResults = pickle.load(f)
    return model_fit


def forecast_future_arima(
    model_fit: ARIMAResults,
    steps: int,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Forecast future 'steps' periods using statsmodels ARIMAResults.
    Returns DataFrame with columns: ['mean', 'lower', 'upper'] indexed by forecast dates.
    """
    fc = model_fit.get_forecast(steps=steps)
    mean = fc.predicted_mean
    conf = fc.conf_int(alpha=alpha)

    # Standardize column names to ['lower','upper']
    if isinstance(conf.columns, pd.MultiIndex):
        lower = conf.iloc[:, 0]
        upper = conf.iloc[:, 1]
    else:
        # handle common names like 'lower y' / 'upper y'
        cols = conf.columns.str.lower()
        lower_col = cols[0]
        upper_col = cols[1]
        lower = conf.iloc[:, 0]
        upper = conf.iloc[:, 1]

    out = pd.DataFrame(
        {"mean": mean.values, "lower": lower.values, "upper": upper.values},
        index=mean.index
    )
    out.index.name = "Date"
    return out


def plot_future_forecast_arima(
    hist_series: pd.Series,
    fc_df: pd.DataFrame,
    title: str = "TSLA – ARIMA 6–12M Forecast (with 95% CI)"
):
    plt.figure(figsize=(12,6))
    plt.plot(hist_series.index, hist_series.values, label="Historical")
    plt.plot(fc_df.index, fc_df["mean"].values, label="ARIMA Forecast")
    plt.fill_between(fc_df.index, fc_df["lower"].values, fc_df["upper"].values,
                     alpha=0.2, label="95% CI")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -------------------------------
# LSTM – Load & Forecast
# -------------------------------

def load_lstm_and_scaler(model_path: str, scaler_path: str):
    """Load Keras model and corresponding MinMaxScaler (joblib)."""
    model = load_model(model_path)
    scaler = joblib_load(scaler_path)
    return model, scaler


def _recursive_lstm_point_forecast(
    last_values: np.ndarray,  # shape (T, 1) raw prices (not scaled)
    model,
    scaler,
    lookback: int,
    steps: int
) -> np.ndarray:
    """
    Deterministic recursive forecast for LSTM (no uncertainty).
    - Scale with scaler, roll the window, feed-forward, inverse-transform.
    Returns array of shape (steps,).
    """
    hist = last_values.copy().reshape(-1, 1)  # raw
    preds = []

    # start with scaled history
    hist_scaled = scaler.transform(hist)

    for _ in range(steps):
        window = hist_scaled[-lookback:]  # shape (lookback, 1)
        x = window.reshape(1, lookback, 1)
        yhat_scaled = model.predict(x, verbose=0)
        yhat = scaler.inverse_transform(yhat_scaled)[0, 0]
        preds.append(yhat)

        # append to hist for next step (both raw and scaled track)
        hist = np.vstack([hist, [[yhat]]])
        hist_scaled = scaler.transform(hist)

    return np.array(preds)


def _mc_dropout_lstm_forecast(
    last_values: np.ndarray,
    model,
    scaler,
    lookback: int,
    steps: int,
    passes: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo dropout inference:
    - Runs multiple stochastic forward passes with `training=True`
    - Builds an empirical distribution for each horizon step
    Returns: (mean_preds, std_preds)
    """
    hist_raw = last_values.copy().reshape(-1, 1)
    # Pre-scale full history (we will re-transform after each append)
    hist_scaled = scaler.transform(hist_raw)

    all_paths = np.zeros((passes, steps), dtype=float)

    for k in range(passes):
        hist_scaled_k = hist_scaled.copy()
        preds_k = []

        for t in range(steps):
            window = hist_scaled_k[-lookback:]
            x = window.reshape(1, lookback, 1)

            # Enable dropout during inference:
            yhat_scaled = model(x, training=True).numpy()  # (1,1)
            yhat = scaler.inverse_transform(yhat_scaled)[0, 0]
            preds_k.append(yhat)

            # append prediction and rescale entire hist_raw -> hist_scaled_k
            # (keeps scaler consistent with training fit)
            hist_raw = np.vstack([hist_raw, [[yhat]]])
            hist_scaled_k = scaler.transform(hist_raw)

        all_paths[k, :] = preds_k

        # reset hist_raw for next MC path
        hist_raw = last_values.copy().reshape(-1, 1)

    mean_preds = all_paths.mean(axis=0)
    std_preds = all_paths.std(axis=0)
    return mean_preds, std_preds


def forecast_future_lstm(
    hist_series: pd.Series,
    model,
    scaler,
    lookback: int = 60,
    steps: int = 126,         # ~6 months of business days
    use_mc_dropout: bool = True,
    mc_passes: int = 200,
    ci_alpha: float = 0.05
) -> pd.DataFrame:
    """
    LSTM future forecast:
      - If use_mc_dropout=True: compute mean and quantile CI via MC sampling.
      - Else: do deterministic forecast and use residual-based CI as a fallback.
    Returns DataFrame ['mean','lower','upper'] indexed by business dates ahead.
    """

    # last known raw values
    last_values = hist_series.values.reshape(-1, 1)

    if use_mc_dropout:
        mean_preds, std_preds = _mc_dropout_lstm_forecast(
            last_values=last_values,
            model=model,
            scaler=scaler,
            lookback=lookback,
            steps=steps,
            passes=mc_passes
        )
        # approximate normal CI from MC mean/std
        z = abs(float(pd.Series([ci_alpha/2]).pipe(lambda s: s.apply(lambda x: 0)).shape[0]))  # not used
        # use empirical quantiles instead (more robust for MC):
        # run MC again to keep memory light? We already have all_paths only as mean/std.
        # Simple quantile approx: mean ± 1.96*std
        # (if you want exact quantiles, modify _mc_dropout_lstm_forecast to return all_paths)
        lower = mean_preds - 1.96 * std_preds
        upper = mean_preds + 1.96 * std_preds

    else:
        # Deterministic forecast
        mean_preds = _recursive_lstm_point_forecast(
            last_values=last_values,
            model=model,
            scaler=scaler,
            lookback=lookback,
            steps=steps
        )

        # Residual-based CI from the last 252 days (1Y) of historical returns
        returns = pd.Series(hist_series).pct_change().dropna()
        resid_std = returns.tail(252).std() if len(returns) >= 20 else returns.std()
        # translate return std to price band (approx): price_t * (1 ± 1.96*std)
        # This is a rough approximation; MC dropout is preferred.
        lower = mean_preds * (1 - 1.96 * resid_std)
        upper = mean_preds * (1 + 1.96 * resid_std)

    future_idx = _future_bdays(hist_series.index[-1], steps)
    out = pd.DataFrame({"mean": mean_preds, "lower": lower, "upper": upper}, index=future_idx)
    out.index.name = "Date"
    return out


def plot_future_forecast_lstm(
    hist_series: pd.Series,
    fc_df: pd.DataFrame,
    title: str = "TSLA – LSTM 6–12M Forecast (with CI)"
):
    plt.figure(figsize=(12,6))
    plt.plot(hist_series.index, hist_series.values, label="Historical")
    plt.plot(fc_df.index, fc_df["mean"].values, label="LSTM Forecast")
    plt.fill_between(fc_df.index, fc_df["lower"].values, fc_df["upper"].values,
                     alpha=0.2, label="Confidence Interval")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
