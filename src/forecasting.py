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

# ARIMA
from pmdarima import auto_arima

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

class ArimaResult:
    preds: pd.Series
    conf_int: pd.DataFrame
    metrics: dict


def fit_arima_auto(y_train: pd.Series):
    model = auto_arima(
        y_train,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        information_criterion="aic",
        error_action="ignore",
        max_p=6,
        max_q=6,
        max_d=2,
    )
    return model


def forecast_arima(model, y_train: pd.Series, y_test: pd.Series) -> ArimaResult:
    n_periods = len(y_test)
    fc, conf = model.predict(n_periods=n_periods, return_conf_int=True)
    preds = pd.Series(fc, index=y_test.index, name="ARIMA_Forecast")
    conf_df = pd.DataFrame(conf, index=y_test.index, columns=["lower", "upper"])
    metrics = evaluate_forecast(y_test.values, preds.values)
    return ArimaResult(preds=preds, conf_int=conf_df, metrics=metrics)


def plot_arima(y_train, y_test, arima_res: ArimaResult, title="TSLA ARIMA Forecast"):
    plt.figure(figsize=(12,6))
    plt.plot(y_train.index, y_train.values, label="Train")
    plt.plot(y_test.index, y_test.values, label="Test", alpha=0.8)
    plt.plot(arima_res.preds.index, arima_res.preds.values, label="ARIMA Forecast")
    plt.fill_between(
        arima_res.conf_int.index,
        arima_res.conf_int["lower"],
        arima_res.conf_int["upper"],
        alpha=0.2, label="95% CI"
    )
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()