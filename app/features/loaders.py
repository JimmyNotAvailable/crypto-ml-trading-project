# loaders.py
# Module tải dữ liệu thô và gọi các hàm chuyên biệt

import os
import pandas as pd

# === Import các hàm chuyên biệt từ các file chức năng ===
from preprocess_aggregate import clean_crypto_data
from feature_engineering import create_features
from binance_api import fetch_binance_ohlcv
from series import resample_ohlcv


def project_root_path() -> str:
    """
    Lấy đường dẫn gốc của project (crypto-project/)
    loaders.py đang ở: app/features/loaders.py -> lùi 2 cấp.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Tải dữ liệu thô từ file CSV
    """
    df = pd.read_csv(file_path)
    print(f"Đã tải dữ liệu từ: {file_path}, số dòng: {len(df)}")
    return df


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hoá:
      - Tạo cột thời gian 'timestamp' từ 'date' (nếu chưa có).
      - Chuẩn tên cột OHLCV về dạng thường: open, high, low, close, volume.
      - Cho phép các biến thể: 'volume usdt' -> 'volume', 'Volume USDT' -> 'volume', ...
    """
    df = df.copy()

    # Chuẩn cột thời gian
    if 'timestamp' not in df.columns:
        if 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
        else:
            # Không có date/timestamp -> để series.resample_ohlcv báo lỗi rõ ràng
            pass
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Map tên cột về chuẩn thường
    col_map = {}
    cols_lower = {c.lower(): c for c in df.columns}  # map lower->original

    # Các tên có thể gặp
    candidates = {
        'open': ['open', 'Open'],
        'high': ['high', 'High'],
        'low': ['low', 'Low'],
        'close': ['close', 'Close'],
        'volume': ['volume', 'Volume', 'volume usdt', 'Volume USDT'],
        'symbol': ['symbol', 'Symbol'],
    }

    for std, alts in candidates.items():
        for name in alts:
            if name in df.columns:
                col_map[name] = std
                break
        else:
            # Không tìm thấy biến thể nào -> bỏ qua
            pass

    if col_map:
        df = df.rename(columns=col_map)

    return df


def save_to_cache(df: pd.DataFrame, filename: str = "crypto_data_cleaned_test.csv"):
    """
    Lưu file vào crypto-project/data/cache
    """
    root = project_root_path()
    cache_path = os.path.join(root, "data", "cache")
    os.makedirs(cache_path, exist_ok=True)
    out_path = os.path.join(cache_path, filename)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Đã lưu file kiểm thử vào: {out_path}")


if __name__ == "__main__":
    # 1) Đọc & tiền xử lý tổng hợp
    root = project_root_path()
    raw_path = os.path.join(root, "data", "ohlcv", "crypto_data.csv")

    df_raw = load_csv(raw_path)
    df_clean = clean_crypto_data(raw_path)

    print("Kết quả sau tiền xử lý:")
    print(df_clean.info())
    print(df_clean.head())
    print("Các trường trong dữ liệu:", df_clean.columns.tolist())
    print("Số dòng dữ liệu sau xử lý:", len(df_clean))

    # 2) Normalize columns và tạo features
    df_clean = normalize_ohlcv_columns(df_clean)
    features = create_features(df_clean)  # Tạo MA, volatility, returns
    
    # 3) Lưu bản với features để kiểm thử
    save_to_cache(features, filename="crypto_data_with_features.csv")
    
    # 4) Resample OHLCV (ngày) với features
    df_resampled = resample_ohlcv(features, rule='D')
    save_to_cache(df_resampled, filename="crypto_data_resampled_D.csv")

    # 5) Lấy thử data từ Binance (minh hoạ)
    df_binance = fetch_binance_ohlcv(symbol='BTCUSDT', interval='1h', limit=100)
    if df_binance is not None and len(df_binance) > 0:
        df_binance_norm = normalize_ohlcv_columns(df_binance)
        df_binance_features = create_features(df_binance_norm)
        save_to_cache(df_binance_features, filename="binance_btcusdt_1h_with_features.csv")

    print("Hoàn tất các bước xử lý dữ liệu với features!")
