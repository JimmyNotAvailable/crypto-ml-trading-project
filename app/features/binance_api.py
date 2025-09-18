# binance_api.py
# Lấy dữ liệu OHLCV từ Binance API cho nhiều coin

# binance_api.py
# Lấy dữ liệu OHLCV từ Binance API cho nhiều coin

import requests
import pandas as pd

def fetch_binance_ohlcv(symbol='BTCUSDT', interval='1h', limit=1000):
    """
    Lấy dữ liệu OHLCV cho một symbol từ Binance API
    symbol: ví dụ 'BTCUSDT'
    interval: ví dụ '1h', '1d', '5m', ...
    limit: số lượng nến lấy về
    Trả về DataFrame OHLCV
    """
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        'open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'close_time', 'Quote_asset_volume', 'Number_of_trades',
        'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'
    ])
    # Convert to numeric types
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    df['symbol'] = symbol
    print(f"Đã lấy dữ liệu OHLCV từ Binance cho {symbol}")
    return df

def fetch_multi_binance_ohlcv(symbols, interval='1h', limit=1000):
    """
    Lấy dữ liệu OHLCV cho nhiều symbol từ Binance API
    symbols: list các symbol, ví dụ ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    Trả về dict: {symbol: DataFrame}
    """
    results = {}
    for symbol in symbols:
        try:
            df = fetch_binance_ohlcv(symbol, interval, limit)
            results[symbol] = df
        except Exception as e:
            print(f"Lỗi lấy dữ liệu cho {symbol}: {e}")
    return results

# # Ví dụ sử dụng: lấy dữ liệu cho nhiều coin
# if __name__ == "__main__":
#     coin_list = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']
#     data_dict = fetch_multi_binance_ohlcv(coin_list, interval='1h', limit=500)
#     for symbol, df in data_dict.items():
#         print(f"{symbol}: {len(df)} dòng dữ liệu, cột: {df.columns.tolist()}")
