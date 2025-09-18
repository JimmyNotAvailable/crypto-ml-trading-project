# feature_engineering.py
# Trích xuất đặc trưng, tạo feature cho ML từ dữ liệu OHLCV

import pandas as pd

def create_features(df):
    # Tạo moving average, volatility, returns cho cột 'close'
    df['ma_10'] = df['close'].rolling(window=10).mean()
    df['ma_50'] = df['close'].rolling(window=50).mean()
    df['volatility'] = df['close'].rolling(window=10).std()
    df['returns'] = df['close'].pct_change()
    print("Đã tạo các feature: ma_10, ma_50, volatility, returns")
    return df
 