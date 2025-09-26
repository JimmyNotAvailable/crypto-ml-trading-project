#!/usr/bin/env python3
"""Debug script to examine dataset and model behavior."""

import pickle
import pandas as pd
import sys
import os

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def examine_dataset():
    """Examine the cached dataset used for training."""
    print("=== Dataset Analysis ===")
    
    with open('data/cache/ml_datasets_top3.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print('Dataset keys:', list(data.keys()))
    
    train = data.get('train_df', pd.DataFrame())
    print('Train shape:', train.shape)
    print('Train columns:', list(train.columns))
    
    if 'target_price' in train.columns:
        print('\nTarget price stats:')
        print(train['target_price'].describe())
        print('\nPrice range:', train['target_price'].min(), 'to', train['target_price'].max())
    
    # Show sample feature values
    feature_cols = [col for col in train.columns if col not in ['date', 'symbol', 'target_price', 'target_price_change', 'target_trend']]
    print('\nFeature columns:', feature_cols)
    
    if len(train) > 0:
        print('\nSample row (features):')
        sample = train[feature_cols].iloc[0]
        for col, val in sample.items():
            print(f"  {col}: {val}")

def test_prediction():
    """Test prediction with realistic vs bot features."""
    print("\n=== Prediction Test ===")
    
    from data.models_production.quick_loader import predict_price
    
    # Bot's current feature generation (scaled to BTC ~117k)
    bot_features = {
        "open": 117212.00 * 0.998,  # 116,978
        "high": 117212.00 * 1.002,  # 117,446
        "low": 117212.00 * 0.995,   # 116,626
        "close": 117212.00,         # 117,212
        "volume": 1_500_000,
        "ma_10": 117212.00 * 0.996, # 116,743
        "ma_50": 117212.00 * 0.97,  # 113,695
        "volatility": 0.02,
        "returns": 0.001,
        "hour": 12,
    }
    
    print("Bot features:")
    for k, v in bot_features.items():
        print(f"  {k}: {v}")
    
    result = predict_price(bot_features)
    print("\nPrediction result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    
    # Try features more similar to training data scale
    print("\n--- Testing with training-scale features ---")
    
    # From dataset analysis, if prices were much lower during training
    training_scale_features = {
        "open": 50000 * 0.998,
        "high": 50000 * 1.002, 
        "low": 50000 * 0.995,
        "close": 50000,
        "volume": 1_500_000,
        "ma_10": 50000 * 0.996,
        "ma_50": 50000 * 0.97,
        "volatility": 0.02,
        "returns": 0.001,
        "hour": 12,
    }
    
    result2 = predict_price(training_scale_features)
    print("Training-scale prediction:")
    for k, v in result2.items():
        print(f"  {k}: {v}")

if __name__ == '__main__':
    examine_dataset()
    test_prediction()