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

# LSTM (TensorFlow Keras)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


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


# LSTM Model

def make_sequences(series_scaled: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(series_scaled)):
        X.append(series_scaled[i - lookback:i])
        y.append(series_scaled[i])
    X = np.array(X)
    y = np.array(y)
    return X.reshape((X.shape[0], X.shape[1], 1)), y.reshape(-1, 1)


def split_sequences_by_index(dates: pd.DatetimeIndex, lookback: int, split_date: str):
    target_dates = dates[lookback:]
    train_mask = target_dates <= pd.to_datetime(split_date)
    if train_mask.any():
        last_train_idx = np.where(train_mask)[0][-1]
        if target_dates[last_train_idx] == pd.to_datetime(split_date):
            train_mask[last_train_idx] = False
    test_mask = ~train_mask
    return train_mask, test_mask, target_dates


def build_lstm(lookback=60, units=64, dropout=0.2, lr=1e-3):
    model = Sequential([
        LSTM(units, input_shape=(lookback, 1), return_sequences=False),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model
@dataclass
class LstmResult:
    preds: pd.Series
    metrics: dict
    history: dict


def train_predict_lstm(
    y_full: pd.Series,
    split_date: str,
    lookback: int = 60,
    epochs: int = 30,
    batch_size: int = 32,
    units: int = 64,
    dropout: float = 0.2,
    lr: float = 1e-3,
    val_split: float = 0.1,
    seed: int = 42,
) -> LstmResult:
    tf.keras.utils.set_random_seed(seed)

    series = y_full.sort_index().astype(float)
    dates = series.index
    values = series.values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(values)

    values_scaled = scaler.transform(values)
    X_all, y_all = make_sequences(values_scaled, lookback=lookback)

    train_mask, test_mask, target_dates = split_sequences_by_index(dates, lookback, split_date)

    train_end_loc = np.where(train_mask)[0][-1] if train_mask.any() else -1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(values[: lookback + train_end_loc + 1])

    model = build_lstm(lookback=lookback, units=units, dropout=dropout, lr=lr)

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        verbose=0,
        callbacks=callbacks
    )

    X_test, y_test_scaled = X_all[test_mask], y_all[test_mask]
    y_pred_scaled = model.predict(X_test, verbose=0)

    y_test = scaler.inverse_transform(y_test_scaled).ravel()
    y_pred = scaler.inverse_transform(y_pred_scaled).ravel()

    test_dates = target_dates[test_mask]
    preds_series = pd.Series(y_pred, index=test_dates, name="LSTM_Forecast")

    metrics = evaluate_forecast(y_test, y_pred)
    return LstmResult(preds=preds_series, metrics=metrics, history=history.history)


def plot_lstm(y_train, y_test, lstm_res: LstmResult, title="TSLA LSTM Forecast"):
    plt.figure(figsize=(12,6))
    plt.plot(y_train.index, y_train.values, label="Train")
    plt.plot(y_test.index, y_test.values, label="Test", alpha=0.8)
    plt.plot(lstm_res.preds.index, lstm_res.preds.values, label="LSTM Forecast")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()