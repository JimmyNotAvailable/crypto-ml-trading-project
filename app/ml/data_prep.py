# data_prep.py
# Chuẩn bị dữ liệu cho ML training: time-series split, feature engineering, target creation

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import os
import sys
from typing import Optional

# Import từ project modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def project_root_path():
    """Lấy đường dẫn root của project"""
    current_file = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

def load_features_data(symbols: Optional[list] = None, top_n: Optional[int] = None) -> pd.DataFrame:
    """
    Load dữ liệu đã có features từ cache - tối ưu cho multiple coins
    
    Args:
        symbols (list): Specific symbols to load
        top_n (int): Load top N symbols by data volume
    """
    root = project_root_path()
    features_path = os.path.join(root, "data", "cache", "crypto_data_with_features.csv")
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file features: {features_path}")
    
    # Load data
    df = pd.read_csv(features_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"🔄 Loaded raw features data: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # Filter by symbols if specified
    if symbols:
        df = df[df['symbol'].isin(symbols)]
        print(f"🔽 Filtered to specified symbols: {len(symbols)} coins, {len(df):,} rows")
    elif top_n:
        # Get top N symbols by data volume
        symbol_counts = df['symbol'].value_counts()
        top_symbols = symbol_counts.head(top_n).index.tolist()
        df = df[df['symbol'].isin(top_symbols)]
        print(f"🔽 Filtered to top {top_n} symbols: {top_symbols}")
        print(f"🔽 Result: {len(df):,} rows")
    
    print(f"📊 Final dataset:")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Symbols: {df['symbol'].nunique()} unique cryptos")
    print(f"  Total records: {len(df):,}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    return df

def create_target_variables(df):
    """
    Tạo target variables cho các loại ML models
    """
    df = df.copy()
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # 1. Price prediction target (next close price)
    df['target_price'] = df.groupby('symbol')['close'].shift(-1)
    
    # 2. Price change percentage target
    df['target_price_change'] = df.groupby('symbol')['close'].pct_change().shift(-1)
    
    # 3. Trend classification target (up/down)
    df['target_trend'] = (df['target_price_change'] > 0).astype(int)
    
    # 4. Volatility level target (high/low volatility)
    df['volatility_rank'] = df.groupby('symbol')['volatility'].rank(pct=True)
    df['target_volatility_level'] = (df['volatility_rank'] > 0.7).astype(int)  # Top 30% = high volatility
    
    print("Created target variables:")
    print("  - target_price: Next close price")
    print("  - target_price_change: Next price change %")
    print("  - target_trend: Binary trend (0=down, 1=up)")
    print("  - target_volatility_level: Binary volatility (0=low, 1=high)")
    
    return df

def prepare_features(df, feature_cols=None):
    """
    Chuẩn bị feature matrix để training
    """
    if feature_cols is None:
        # Default features for ML
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'ma_10', 'ma_50', 'volatility', 'returns',
            'hour'  # Time feature
        ]
    
    # Chỉ lấy các cột tồn tại
    available_features = [col for col in feature_cols if col in df.columns]
    print(f"Using features: {available_features}")
    
    # Cần giữ lại các cột cần thiết cho split và metadata
    required_cols = available_features + ['target_price', 'target_price_change', 'target_trend', 'date', 'symbol']
    available_required = [col for col in required_cols if col in df.columns]
    
    # Remove rows with NaN in features or targets
    feature_df = df[available_required].dropna(subset=available_features + ['target_price', 'target_price_change', 'target_trend'])
    
    X = feature_df[available_features]
    targets = {
        'price': feature_df['target_price'],
        'price_change': feature_df['target_price_change'], 
        'trend': feature_df['target_trend']
    }
    
    print(f"Feature matrix: {X.shape}")
    print(f"Valid samples after removing NaN: {len(feature_df)}")
    
    return X, targets, feature_df

def time_series_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Chia dữ liệu theo thời gian để tránh data leakage
    train_ratio + val_ratio + test_ratio = 1.0
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    
    df_sorted = df.sort_values('date').reset_index(drop=True)
    n = len(df_sorted)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df_sorted.iloc[:train_end]
    val_df = df_sorted.iloc[train_end:val_end]
    test_df = df_sorted.iloc[val_end:]
    
    print("Time series split:")
    print(f"  Train: {len(train_df)} samples ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"  Val:   {len(val_df)} samples ({val_df['date'].min()} to {val_df['date'].max()})")
    print(f"  Test:  {len(test_df)} samples ({test_df['date'].min()} to {test_df['date'].max()})")
    
    return train_df, val_df, test_df

def scale_features(X_train, X_val, X_test, scaler_type='standard'):
    """
    Scale features using training data statistics
    """
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type must be 'standard' or 'minmax'")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Applied {scaler_type} scaling to features")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def prepare_ml_datasets(symbol_filter=None, feature_cols=None, scaler_type='standard'):
    """
    Toàn bộ pipeline chuẩn bị dữ liệu cho ML
    symbol_filter: list of symbols to include (None = all symbols)
    """
    print("=== PREPARING ML DATASETS ===")
    
    # 1. Load data
    df = load_features_data()
    
    # 2. Filter symbols if specified
    if symbol_filter:
        df = df[df['symbol'].isin(symbol_filter)]
        print(f"Filtered to symbols: {symbol_filter}")
        print(f"Remaining data: {len(df)} rows")
    
    # 3. Create targets
    df = create_target_variables(df)
    
    # 4. Prepare features
    X, targets, feature_df = prepare_features(df, feature_cols)
    
    # 5. Time series split
    train_df, val_df, test_df = time_series_split(feature_df)
    
    # 6. Split features and targets
    feature_cols_final = X.columns.tolist()
    
    X_train = train_df[feature_cols_final]
    X_val = val_df[feature_cols_final]
    X_test = test_df[feature_cols_final]
    
    y_train = {
        'price': train_df['target_price'],
        'price_change': train_df['target_price_change'],
        'trend': train_df['target_trend']
    }
    y_val = {
        'price': val_df['target_price'],
        'price_change': val_df['target_price_change'],
        'trend': val_df['target_trend']
    }
    y_test = {
        'price': test_df['target_price'],
        'price_change': test_df['target_price_change'],
        'trend': test_df['target_trend']
    }
    
    # 7. Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, X_test, scaler_type
    )
    
    # 8. Package results
    datasets = {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'X_train_raw': X_train,
        'X_val_raw': X_val,
        'X_test_raw': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler,
        'feature_cols': feature_cols_final,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df
    }
    
    print("\n=== DATASET PREPARATION COMPLETE ===")
    print(f"Features: {len(feature_cols_final)}")
    print(f"Train samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    return datasets

def save_prepared_datasets(datasets, filename_prefix='ml_datasets'):
    """
    Lưu prepared datasets vào cache
    """
    root = project_root_path()
    cache_path = os.path.join(root, "data", "cache")
    
    # Save datasets to pickle for easy loading
    import pickle
    
    datasets_path = os.path.join(cache_path, f"{filename_prefix}.pkl")
    with open(datasets_path, 'wb') as f:
        pickle.dump(datasets, f)
    
    print(f"Saved prepared datasets to: {datasets_path}")
    
    # Also save feature names as text for reference
    feature_path = os.path.join(cache_path, f"{filename_prefix}_features.txt")
    with open(feature_path, 'w') as f:
        f.write("ML Dataset Features:\n")
        f.write("===================\n\n")
        for i, feature in enumerate(datasets['feature_cols']):
            f.write(f"{i+1}. {feature}\n")
        
        f.write(f"\nDataset sizes:\n")
        f.write(f"Train: {len(datasets['X_train'])}\n")
        f.write(f"Val: {len(datasets['X_val'])}\n")
        f.write(f"Test: {len(datasets['X_test'])}\n")
    
    print(f"Saved feature info to: {feature_path}")
    return datasets_path

def load_prepared_datasets(filename_prefix='ml_datasets'):
    """
    Load prepared datasets từ cache
    """
    root = project_root_path()
    cache_path = os.path.join(root, "data", "cache")
    datasets_path = os.path.join(cache_path, f"{filename_prefix}.pkl")
    
    if not os.path.exists(datasets_path):
        raise FileNotFoundError(f"Không tìm thấy prepared datasets: {datasets_path}")
    
    import pickle
    with open(datasets_path, 'rb') as f:
        datasets = pickle.load(f)
    
    print(f"Loaded prepared datasets from: {datasets_path}")
    print(f"Features: {len(datasets['feature_cols'])}")
    print(f"Train: {len(datasets['X_train'])}, Val: {len(datasets['X_val'])}, Test: {len(datasets['X_test'])}")
    
    return datasets

if __name__ == "__main__":
    # Configurable dataset preparation - optimized for scalability
    try:
        print("🚀 ML DATA PREPARATION PIPELINE")
        
        # Configuration options:
        use_top_n = True  # Set False to use specific symbols
        top_n = 3  # Change to 34 for full dataset
        specific_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT']
        
        if use_top_n:
            print(f"📊 Preparing datasets for TOP {top_n} coins...")
            df = load_features_data()
            dataset_name = f'ml_datasets_top{top_n}'
        else:
            print(f"📊 Preparing datasets for specific coins: {specific_symbols}")
            df = load_features_data(symbols=specific_symbols) 
            dataset_name = 'ml_datasets_custom'
        
        # Prepare datasets
        datasets = prepare_ml_datasets(df)
        
        # Save for later use
        cache_path = save_prepared_datasets(datasets, dataset_name)
        
        print(f"\n✅ PREPARATION COMPLETE!")
        print(f"📁 Saved to: {cache_path}")
        print(f"🎯 Ready for ML model training!")
        print(f"📊 {datasets['metadata']['n_symbols']} symbols processed")
        
    except Exception as e:
        print(f"❌ Error in data preparation: {e}")
        import traceback
        traceback.print_exc()