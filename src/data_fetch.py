# Data Fetching Module For GMF Interim Submission

import os
import yfinance as yf
import pandas as pd

DATA_DIR = "data"

def fetch_asset_data(ticker, start, end):
    """
    Download historical data for a given ticker from Yahoo Finance.
    Saves CSV to data/ directory.
    """
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, start=start, end=end, progress=False)

    # Ensure clean structure
    df.index = pd.to_datetime(df.index)
    df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

    # Save to CSV
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    df.to_csv(file_path)
    print(f"✅ Saved {ticker} data to {file_path}")

    return df
