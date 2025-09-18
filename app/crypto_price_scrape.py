# crypto_price_scrape.py
# Thu thập dữ liệu giá crypto từ các nguồn

import os
import pandas as pd

def load_all_ohlcv_data(data_dir='../data/ohlcv'):
	"""
	Đọc toàn bộ file CSV trong thư mục data/ohlcv và trả về một DataFrame tổng hợp
	"""
	all_dfs = []
	for filename in os.listdir(data_dir):
		if filename.endswith('.csv'):
			file_path = os.path.join(data_dir, filename)
			try:
				df = pd.read_csv(file_path)
				df['symbol'] = filename.split('_')[1] if filename.startswith('Binance_') else filename.replace('.csv', '')
				all_dfs.append(df)
				print(f"Đã đọc: {filename}, số dòng: {len(df)}")
			except Exception as e:
				print(f"Lỗi đọc {filename}: {e}")
	if all_dfs:
		combined_df = pd.concat(all_dfs, ignore_index=True)
		print(f"Tổng số dòng dữ liệu: {len(combined_df)}")
		return combined_df
	else:
		print("Không tìm thấy file dữ liệu nào.")
		return None

if __name__ == "__main__":
	df = load_all_ohlcv_data('data/ohlcv')
	if df is not None:
		print(df.head())
		print(df.info())
		print(df.describe())
		print(df['symbol'].unique())
