# data_analysis.py
# Phân tích dữ liệu exploratory để hiểu dataset crypto

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Import từ features module - sửa lỗi import path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def project_root_path():
    """
    Lấy đường dẫn root của project
    """
    current_file = os.path.abspath(__file__)
    # Từ app/ml/data_analysis.py -> project root
    return os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

def load_cleaned_data(sample_size=None, symbols=None):
    """
    Load dữ liệu đã cleaned từ cache - tối ưu cho 34 coins
    
    Args:
        sample_size (int): Giới hạn số dòng để tiết kiệm memory
        symbols (list): Chỉ load symbols cụ thể
    """
    root = project_root_path()
    
    # Ưu tiên file có features
    features_path = os.path.join(root, "data", "cache", "crypto_data_with_features.csv")
    cache_path = os.path.join(root, "data", "cache", "crypto_data_cleaned_test.csv")
    
    if os.path.exists(features_path):
        # Đọc với chunks để tối ưu memory cho 34 coins
        print(f"Loading data with features...")
        df = pd.read_csv(features_path, 
                        dtype={'open': 'float32', 'high': 'float32', 'low': 'float32', 
                               'close': 'float32', 'volume': 'float32'})
        print(f"✅ Loaded WITH FEATURES: {len(df)} dòng, {len(df.columns)} cột")
    elif os.path.exists(cache_path):
        df = pd.read_csv(cache_path,
                        dtype={'open': 'float32', 'high': 'float32', 'low': 'float32', 
                               'close': 'float32'})
        print(f"✅ Loaded WITHOUT features: {len(df)} dòng, {len(df.columns)} cột")
    else:
        raise FileNotFoundError(f"❌ Không tìm thấy file dữ liệu trong cache")
    
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter symbols nếu được chỉ định
    if symbols:
        df = df[df['symbol'].isin(symbols)]
        print(f"🔽 Filtered to {len(symbols)} symbols: {len(df)} dòng")
    
    # Sample data nếu cần
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).sort_values(['symbol', 'date'])
        print(f"🔽 Sampled to {sample_size} dòng")
    
    print(f"📊 Final dataset: {df['symbol'].nunique()} symbols, {len(df)} records")
    print(f"💾 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    return df

def basic_info_analysis(df):
    """
    Phân tích thông tin cơ bản về dataset - tối ưu cho 34 coins
    """
    print("=== THÔNG TIN CƠ BẢN ===")
    print(f"📐 Kích thước dataset: {df.shape}")
    print(f"📅 Khoảng thời gian: {df['date'].min()} đến {df['date'].max()}")
    print(f"🪙 Số crypto unique: {df['symbol'].nunique()}")
    
    # Top 10 symbols và thống kê nhanh
    symbol_counts = df['symbol'].value_counts()
    print(f"📊 Top 10 crypto theo số dòng dữ liệu:")
    print(symbol_counts.head(10))
    
    # Data quality check
    print(f"\n📈 Data quality:")
    print(f"  Min records per symbol: {symbol_counts.min()}")
    print(f"  Max records per symbol: {symbol_counts.max()}")
    print(f"  Avg records per symbol: {symbol_counts.mean():.0f}")
    
    print("\n⚠️  MISSING VALUES:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_info = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })
    missing_info = missing_info[missing_info['Missing Count'] > 0]
    
    if len(missing_info) > 0:
        print(missing_info)
    else:
        print("✅ Không có missing values!")
    
    return df

def correlation_analysis(df):
    """
    Phân tích correlation matrix giữa các features số - tối ưu cho 34 coins
    """
    print("\n=== CORRELATION ANALYSIS ===")
    
    # Auto-detect available columns
    base_cols = []
    for col in ['open', 'high', 'low', 'close', 'tradecount']:
        if col in df.columns:
            base_cols.append(col)
    
    # Add volume với tên linh hoạt
    volume_cols = ['volume usdt', 'volume', 'volume_base', 'volume_quote']
    for vol_col in volume_cols:
        if vol_col in df.columns:
            base_cols.append(vol_col)
            break
    
    # Features engineering nếu có
    feature_cols = ['ma_10', 'ma_50', 'volatility', 'returns', 'rsi', 'bb_upper', 'bb_lower']
    available_features = [col for col in feature_cols if col in df.columns]
    
    numeric_cols = base_cols + available_features
    
    if not numeric_cols:
        print("❌ Không tìm thấy cột số nào để phân tích correlation!")
        return df
    
    print(f"✅ Analyzing {len(numeric_cols)} columns: {numeric_cols}")
    
    # Sample data nếu quá lớn (để tăng tốc visualization)
    if len(df) > 50000:
        df_sample = df.sample(n=50000, random_state=42)
        print(f"🔽 Sampled {len(df_sample)} rows for correlation analysis")
    else:
        df_sample = df
    
    # Calculate correlation matrix
    corr_matrix = df_sample[numeric_cols].corr()
    
    # Visualize correlation matrix
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Hide upper triangle
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, fmt='.2f', 
                cbar_kws={"shrink": .8})
    plt.title(f'Correlation Matrix - Crypto Features ({df["symbol"].nunique()} coins)')
    plt.tight_layout()
    
    # Lưu biểu đồ
    root = project_root_path()
    plot_path = os.path.join(root, "data", "cache", "correlation_matrix.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Đã lưu correlation matrix: {plot_path}")
    plt.show()
    
    return corr_matrix

def price_trend_analysis(df, sample_symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT']):
    """
    Visualize price trends cho một số crypto chính
    """
    print(f"\n=== PRICE TREND ANALYSIS ===")
    
    available_symbols = df['symbol'].unique()
    plot_symbols = [s for s in sample_symbols if s in available_symbols]
    
    if not plot_symbols:
        plot_symbols = available_symbols[:3]  # Lấy 3 symbol đầu tiên
    
    print(f"Phân tích trend cho: {plot_symbols}")
    
    fig, axes = plt.subplots(len(plot_symbols), 1, figsize=(15, 5*len(plot_symbols)))
    if len(plot_symbols) == 1:
        axes = [axes]
    
    for i, symbol in enumerate(plot_symbols):
        symbol_data = df[df['symbol'] == symbol].sort_values('date')
        
        axes[i].plot(symbol_data['date'], symbol_data['close'], label='Close Price', alpha=0.8)
        
        # Thêm moving averages nếu có
        if 'ma_10' in symbol_data.columns:
            axes[i].plot(symbol_data['date'], symbol_data['ma_10'], label='MA 10', alpha=0.7)
        if 'ma_50' in symbol_data.columns:
            axes[i].plot(symbol_data['date'], symbol_data['ma_50'], label='MA 50', alpha=0.7)
        
        axes[i].set_title(f'{symbol} - Price Trend')
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel('Price (USDT)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Lưu biểu đồ
    root = project_root_path()
    plot_path = os.path.join(root, "data", "cache", "price_trends.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Đã lưu price trends: {plot_path}")
    plt.show()

def volume_analysis(df):
    """
    Phân tích volume patterns
    """
    print(f"\n=== VOLUME ANALYSIS ===")
    
    # Xác định tên cột volume
    volume_col = 'volume' if 'volume' in df.columns else 'volume usdt'
    
    if volume_col not in df.columns:
        print("Không tìm thấy cột volume, bỏ qua analysis này")
        return
    
    # Volume distribution
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(df[volume_col], bins=50, alpha=0.7, edgecolor='black')
    plt.title('Volume Distribution')
    plt.xlabel(f'{volume_col}')
    plt.ylabel('Frequency')
    plt.yscale('log')
    
    plt.subplot(1, 3, 2)
    plt.hist(np.log10(df[volume_col] + 1), bins=50, alpha=0.7, edgecolor='black')
    plt.title('Log Volume Distribution')
    plt.xlabel(f'Log10({volume_col} + 1)')
    plt.ylabel('Frequency')
    
    # Volume by crypto
    plt.subplot(1, 3, 3)
    top_symbols = df['symbol'].value_counts().head(10).index
    volume_by_symbol = df[df['symbol'].isin(top_symbols)].groupby('symbol')[volume_col].mean()
    volume_by_symbol.plot(kind='bar')
    plt.title('Average Volume by Top 10 Crypto')
    plt.xlabel('Symbol')
    plt.ylabel(f'Average {volume_col}')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Lưu biểu đồ
    root = project_root_path()
    plot_path = os.path.join(root, "data", "cache", "volume_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Đã lưu volume analysis: {plot_path}")
    plt.show()

def feature_distribution_analysis(df):
    """
    Phân tích phân phối các features nếu có
    """
    feature_cols = ['ma_10', 'ma_50', 'volatility', 'returns']
    available_features = [col for col in feature_cols if col in df.columns]
    
    if not available_features:
        print("\n=== FEATURE ANALYSIS ===")
        print("Chưa có features engineering, bỏ qua bước này")
        return
    
    print(f"\n=== FEATURE DISTRIBUTION ANALYSIS ===")
    print(f"Features có sẵn: {available_features}")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(available_features[:4]):
        if i < len(axes):
            # Loại bỏ NaN values
            feature_data = df[feature].dropna()
            
            axes[i].hist(feature_data, bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{feature} Distribution')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
            
            # Thêm stats
            mean_val = feature_data.mean()
            std_val = feature_data.std()
            axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
            axes[i].legend()
    
    # Ẩn axes thừa
    for i in range(len(available_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Lưu biểu đồ
    root = project_root_path()
    plot_path = os.path.join(root, "data", "cache", "feature_distributions.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Đã lưu feature distributions: {plot_path}")
    plt.show()

def generate_summary_report(df, corr_matrix):
    """
    Tạo báo cáo tổng hợp về dataset
    """
    print("\n" + "="*60)
    print("              SUMMARY REPORT")
    print("="*60)
    
    print(f"Dataset size: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Number of unique cryptocurrencies: {df['symbol'].nunique()}")
    
    # Top correlations
    print(f"\nTop positive correlations:")
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append((
                corr_matrix.columns[i], 
                corr_matrix.columns[j], 
                corr_matrix.iloc[i, j]
            ))
    
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for col1, col2, corr in corr_pairs[:5]:
        print(f"  {col1} - {col2}: {corr:.4f}")
    
    # Basic stats
    print(f"\nPrice statistics (close):")
    print(f"  Mean: ${df['close'].mean():.4f}")
    print(f"  Median: ${df['close'].median():.4f}")
    print(f"  Std: ${df['close'].std():.4f}")
    print(f"  Min: ${df['close'].min():.4f}")
    print(f"  Max: ${df['close'].max():.4f}")
    
    # Lưu report
    root = project_root_path()
    report_path = os.path.join(root, "data", "cache", "data_analysis_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("CRYPTO DATA ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"Dataset size: {df.shape[0]:,} rows, {df.shape[1]} columns\n")
        f.write(f"Date range: {df['date'].min()} to {df['date'].max()}\n")
        f.write(f"Cryptocurrencies: {df['symbol'].nunique()}\n\n")
        
        f.write("Top correlations:\n")
        for col1, col2, corr in corr_pairs[:10]:
            f.write(f"  {col1} - {col2}: {corr:.4f}\n")
    
    print(f"\nĐã lưu report: {report_path}")

if __name__ == "__main__":
    # Pipeline phân tích dữ liệu hoàn chỉnh - tối ưu cho 34 coins
    try:
        print("🚀 Bắt đầu phân tích dữ liệu crypto...")
        
        # Option to test with sample for large datasets
        use_sample = True  # Set False để analyze full dataset
        sample_size = 100000 if use_sample else None
        
        # 1. Load dữ liệu với optimization
        df = load_cleaned_data(sample_size=sample_size)
        
        # 2. Phân tích cơ bản
        df = basic_info_analysis(df)
        
        # 3. Correlation analysis
        corr_matrix = correlation_analysis(df)
        
        # 4. Price trend analysis
        price_trend_analysis(df)
        
        # 5. Volume analysis
        volume_analysis(df)
        
        # 6. Feature distribution analysis
        feature_distribution_analysis(df)
        
        # 7. Summary report
        generate_summary_report(df, corr_matrix)
        
        print("\n" + "="*60)
        print("✅ Hoàn tất phân tích dữ liệu!")
        print(f"📊 Analyzed {df['symbol'].nunique()} coins")
        print("📁 Các file kết quả được lưu trong data/cache/")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Lỗi trong quá trình phân tích: {e}")
        import traceback
        traceback.print_exc()
# if __name__ == "__main__":
#     # Uncomment để chạy analysis khi cần
#     pass