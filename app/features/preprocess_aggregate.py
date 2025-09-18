# preprocess_aggregate.py
# Xử lý, làm sạch dữ liệu tổng hợp từ các file lớn như crypto_data.csv, cryptotoken_full.csv

import pandas as pd
import os

def clean_crypto_data(input_file, output_file='crypto_data_cleaned.csv', cache_dir='data/cache'):
    df = pd.read_csv(input_file)
    df = df.dropna()
    df = df.drop_duplicates()
    # Chuyển đổi kiểu dữ liệu thời gian nếu có
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
    # Chuẩn hóa symbol nếu có
    if 'symbol' in df.columns:
        df['symbol'] = df['symbol'].str.upper().str.strip()
    # Lưu ra cache
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cache_path = os.path.join(project_root, cache_dir)
    os.makedirs(cache_path, exist_ok=True)
    file_path = os.path.join(cache_path, output_file)
    df.to_csv(file_path, index=False)
    print(f"Đã lưu dữ liệu tổng hợp sạch vào: {file_path}")
    return df
