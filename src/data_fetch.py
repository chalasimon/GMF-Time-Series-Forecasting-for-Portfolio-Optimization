# Data Fetching Module for GMF Interim Submission

import os
import yfinance as yf
import pandas as pd

DATA_DIR = "../data"
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_asset_data(ticker, start, end):
    """
    Download historical data for a given ticker from Yahoo Finance.
    Handles MultiIndex columns and missing fields.
    Saves CSV to data/ directory.
    """
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        print(f"No data returned for {ticker}. Skipping.")
        return None

    # Flatten MultiIndex columns (if present)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join([str(c) for c in col if c]).strip()
            if isinstance(col, tuple) else col
            for col in df.columns
        ]

    # Remove ticker suffix if it exists (e.g., "Close TSLA" -> "Close")
    df.columns = [col.split(" ")[0] for col in df.columns]

    # Ensure datetime index
    df.index = pd.to_datetime(df.index)

    # Filter only columns that actually exist
    desired_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    existing_cols = [col for col in desired_cols if col in df.columns]
    if not existing_cols:
        print(f"No matching OHLCV columns found for {ticker}. Skipping.")
        return None

    df = df[existing_cols]

    # Save to CSV
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    df.to_csv(file_path)
    print(f" Saved {ticker} data to {file_path}")

    return df