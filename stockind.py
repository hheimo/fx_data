import numpy as np
import pandas as pd


##Get CSV data from path and return pandas dataframe
def getData(fromPath):
    df = pd.read_csv(fromPath,
                     sep=';',
                     names=["Timestamp", "Open", "High", "Low", "Close", "Vol"],
                     index_col=0,
                     parse_dates=True
                     )
    # Deleting redundant data from DF
    del df['Vol']

    # group every 1H to create OHLC if needed
    grouped_data = df.resample('1H').agg({'Open': 'first',
                                          'High': 'max',
                                          'Low': 'min',
                                          'Close': 'last'})

    return df


##Relative Strength Index
def RSI(series, period):
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period - 1]] = np.mean(u[:period])  # first value is sum of avg gains
    u = u.drop(u.index[:(period - 1)])
    d[d.index[period - 1]] = np.mean(d[:period])  # first value is sum of avg losses
    d = d.drop(d.index[:(period - 1)])
    rs = pd.stats.moments.ewma(u, com=period - 1, adjust=False) / \
         pd.stats.moments.ewma(d, com=period - 1, adjust=False)
    return 100 - 100 / (1 + rs)


##Williams %R
def williamsR(high, low, close):
    H = high.rolling(window=14).max()
    L = low.rolling(window=14).min()
    R = (H - close) / (H - L) * -100
    return R


##Stochastics

# Fast %K
def STOK(close, low, high, n):
    STOK = ((close - low.rolling(window=n).min())) / (high.rolling(window=n).max() - low.rolling(window=n).min()) * 100
    return STOK


# Fast %D
def STOD(close, low, high, n):
    STOK = ((close - low.rolling(window=n).min())) / (high.rolling(window=n).max() - low.rolling(window=n).min()) * 100
    STOD = STOK.rolling(window=3).mean()
    return STOD


# Slow %D
def STODS(close, low, high, n):
    STOK = ((close - low.rolling(window=n).min())) / (high.rolling(window=n).max() - low.rolling(window=n).min()) * 100
    STOD = STOK.rolling(window=3).mean()
    STODS = STOD.rolling(window=3).mean()

    return STODS