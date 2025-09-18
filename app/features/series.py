# series.py
# Xử lý chuỗi thời gian OHLCV

import pandas as pd

def resample_ohlcv(df: pd.DataFrame, rule: str = 'D') -> pd.DataFrame:
    """
    Resample dữ liệu OHLCV sang chu kỳ mới (ví dụ: ngày 'D', tuần 'W', tháng 'M', ...).

    Hỗ trợ:
      - Cột thời gian: 'timestamp' hoặc 'date'
      - Tên cột OHLCV dạng thường hoặc hoa:
          open/Open, high/High, low/Low, close/Close, volume/Volume/volume usdt/Volume USDT
    Kết quả trả về dùng tên cột thường: open, high, low, close, volume
    """
    if df is None or len(df) == 0:
        raise ValueError("DataFrame rỗng.")

    df = df.copy()

    # Xác định cột thời gian
    time_col = None
    if 'timestamp' in df.columns:
        time_col = 'timestamp'
    elif 'date' in df.columns:
        time_col = 'date'
    if time_col is None:
        raise ValueError("Thiếu cột thời gian: cần 'timestamp' hoặc 'date'.")

    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col])

    # Map tên cột về chuẩn thường
    name_map = {}
    for std, alts in {
        'open':   ['open', 'Open'],
        'high':   ['high', 'High'],
        'low':    ['low', 'Low'],
        'close':  ['close', 'Close'],
        'volume': ['volume', 'Volume', 'volume usdt', 'Volume USDT'],
    }.items():
        for a in alts:
            if a in df.columns:
                name_map[a] = std
                break

    if name_map:
        df = df.rename(columns=name_map)

    # Chỉ giữ các cột cần thiết nếu tồn tại
    keep_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
    if not keep_cols:
        raise ValueError("Không tìm thấy các cột OHLCV cần thiết.")

    df = df.set_index(time_col)

    agg_dict = {}
    if 'open' in df.columns:   agg_dict['open'] = 'first'
    if 'high' in df.columns:   agg_dict['high'] = 'max'
    if 'low' in df.columns:    agg_dict['low'] = 'min'
    if 'close' in df.columns:  agg_dict['close'] = 'last'
    if 'volume' in df.columns: agg_dict['volume'] = 'sum'

    df_out = df.resample(rule).agg(agg_dict).dropna(how='any').reset_index()
    # Chuẩn tên cột thời gian là 'timestamp' cho thống nhất downstream
    df_out = df_out.rename(columns={time_col: 'timestamp'})

    return df_out
