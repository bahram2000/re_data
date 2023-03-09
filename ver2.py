import yfinance as yf
import pandas as pd
import numpy as np

def getData(ticker, start_date, n=3):
    """
    Scrape historical stock data and create input sequences.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    start_date : str
        Start date for historical data scrape in 'YYYY-MM-DD' format.
    n : int, optional
        Number of consecutive candles to use as input. Default is 3.

    Returns
    -------
    X : np.ndarray
        Input sequences for LSTM model.
    """

    # Scrape data
    df_raw = yf.download(ticker, start=start_date)
    df = df_raw[['Open', 'High', 'Low', 'Close']]

    # Create input sequences (X)
    X = []
    for i in range(n, len(df)):
        row = df[i-n:i].values
        X.append(row / row[0, 0])

    return np.array(X).reshape(-1, n*4)[:,1:]
