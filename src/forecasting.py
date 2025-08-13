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

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

from dataclasses import dataclass

# ARIMA (statsmodels)
from statsmodels.tsa.arima.model import ARIMA


def chronological_split(series: pd.Series, split_date: str):
    """Split a price series into train/test by date string 'YYYY-MM-DD'."""
    series = series.sort_index()
    train = series.loc[:split_date]
    test = series.loc[split_date:]
    # remove the split date from train if duplicated boundary
    if not train.empty and not test.empty and train.index[-1] == test.index[0]:
        test = test.iloc[1:]
    return train, test

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0

def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape_val = mape(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape_val}

@dataclass
class ArimaResult:
    preds: pd.Series
    conf_int: pd.DataFrame
    metrics: dict


def fit_arima(y_train: pd.Series, order=(1,1,1)):
    """Fit ARIMA model using given order (p,d,q)."""
    model = ARIMA(y_train, order=order)
    model_fit = model.fit()
    return model_fit


def forecast_arima(model_fit, y_test: pd.Series) -> ArimaResult:
    n_periods = len(y_test)
    fc_res = model_fit.get_forecast(steps=n_periods)
    preds = pd.Series(fc_res.predicted_mean.values, index=y_test.index, name="ARIMA_Forecast")
    conf_df = fc_res.conf_int()
    metrics = evaluate_forecast(y_test.values, preds.values)
    return ArimaResult(preds=preds, conf_int=conf_df, metrics=metrics)


def plot_arima(y_train, y_test, arima_res, title="TSLA ARIMA Forecast"):
    plt.figure(figsize=(12,6))
    plt.plot(y_train.index, y_train.values, label="Train")
    plt.plot(y_test.index, y_test.values, label="Test", alpha=0.8)
    plt.plot(arima_res.preds.index, arima_res.preds.values, label="ARIMA Forecast")
    
    # statsmodels ARIMA returns conf_int columns as 0 and 1
    conf_lower = arima_res.conf_int.iloc[:, 0]
    conf_upper = arima_res.conf_int.iloc[:, 1]
    
    plt.fill_between(
        arima_res.conf_int.index,
        conf_lower,
        conf_upper,
        alpha=0.2, label="95% CI"
    )
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
