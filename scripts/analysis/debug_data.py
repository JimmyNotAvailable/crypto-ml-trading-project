#!/usr/bin/env python3

import sys
import os
import pickle
import numpy as np
import pandas as pd

# Add correct path to project root (2 levels up from scripts/analysis)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.ml.data_prep import load_prepared_datasets
from app.ml.evaluate import load_trained_models
# Load models
print("Loading models...")
models = load_trained_models()

# Load datasets
print("Loading datasets...")
try:
    with open('data/cache/ml_datasets_top3.pkl', 'rb') as f:
        datasets = pickle.load(f)
except FileNotFoundError:
    print("Error: ml_datasets_top3.pkl not found. Loading from data_prep...")
    datasets = load_prepared_datasets('ml_datasets_top3')

# Debug datasets structure
print("\n=== DATASET STRUCTURE ===")
print("Available keys in datasets:", list(datasets.keys()))

# Check y_train structure
if 'y_train' in datasets:
    y_train = datasets['y_train']
    print(f"y_train type: {type(y_train)}")
    if isinstance(y_train, dict):
        print(f"y_train keys: {list(y_train.keys())}")
    if isinstance(y_train, (pd.DataFrame, np.ndarray)):
        try:
            shape = y_train.shape  # type: ignore[attr-defined]
            print(f"y_train shape: {shape}")
        except Exception:
            pass
    if isinstance(y_train, pd.DataFrame):
        print(f"y_train columns: {y_train.columns.tolist()}")

# Check training data range vs test data range
print("\n=== DATA RANGE COMPARISON ===")
if all(key in datasets for key in ['X_train', 'y_train']):
    print("Training data:")
    X_train = datasets['X_train']
    y_train = datasets['y_train']
    
    # Handle different y_train structures
    if isinstance(y_train, dict) and 'price' in y_train:
        y_train_price = y_train['price']
    elif isinstance(y_train, pd.DataFrame) and 'price' in y_train.columns:
        y_train_price = y_train['price']
    else:
        y_train_price = y_train
    
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_train range: [{X_train.min().min():.4f}, {X_train.max().max():.4f}]")
    try:
        arr = np.asarray(y_train_price)
        print(f"  y_train_price range: [{np.nanmin(arr):.4f}, {np.nanmax(arr):.4f}]")
    except Exception:
        print("  y_train_price range: (unavailable)")

if all(key in datasets for key in ['test_df', 'feature_cols']):
    print("\nTest data from evaluation:")
    test_df = datasets['test_df']
    feature_cols = datasets['feature_cols']
    X_test = test_df[feature_cols].fillna(0)
    y_test_price = test_df['target_price']

    print(f"  X_test shape: {X_test.shape}")
    print(f"  X_test range: [{X_test.min().min():.4f}, {X_test.max().max():.4f}]")
    print(f"  y_test_price range: [{y_test_price.min():.4f}, {y_test_price.max():.4f}]")

    # Check if data is the same
    print(f"\n=== DATA CONSISTENCY CHECK ===")
    if all(key in datasets for key in ['X_test', 'y_test']):
        X_test_direct = datasets['X_test']
        y_test_direct = datasets['y_test']
        
        # Handle different y_test structures
        if isinstance(y_test_direct, dict) and 'price' in y_test_direct:
            y_test_direct = y_test_direct['price']
        elif isinstance(y_test_direct, pd.DataFrame) and 'price' in y_test_direct.columns:
            y_test_direct = y_test_direct['price']

        print(f"Direct from datasets X_test: {X_test_direct.shape}")
        try:
            print(f"Direct from datasets y_test: {np.asarray(y_test_direct).shape}")
        except Exception:
            print("Direct from datasets y_test: (unknown shape)")
        print(f"From test_df X_test: {X_test.shape}")
        print(f"From test_df y_test: {y_test_price.shape}")

        try:
            print(f"\nAre X_test equal? {np.allclose(np.asarray(X_test_direct), X_test.values, equal_nan=True)}")
        except Exception:
            print("\nAre X_test equal? (incomparable shapes or types)")
        try:
            print(f"Are y_test equal? {np.allclose(np.asarray(y_test_direct), np.asarray(y_test_price), equal_nan=True)}")
        except Exception:
            print("Are y_test equal? (incomparable shapes or types)")

        print(f"\nX_test_direct range: [{X_test_direct.min().min() if hasattr(X_test_direct, 'min') else np.min(X_test_direct):.4f}, {X_test_direct.max().max() if hasattr(X_test_direct, 'max') else np.max(X_test_direct):.4f}]")
        try:
            arr_y = np.asarray(y_test_direct)
            print(f"y_test_direct range: [{np.nanmin(arr_y):.4f}, {np.nanmax(arr_y):.4f}]")
        except Exception:
            print("y_test_direct range: (unavailable)")

    # Check for outliers in test data
    print(f"\n=== OUTLIER ANALYSIS ===")
    for col in feature_cols:
        if col in X_test.columns:
            col_data = X_test[col]
            q99 = col_data.quantile(0.99)
            q01 = col_data.quantile(0.01)
            max_val = col_data.max()
            min_val = col_data.min()
            
            if max_val > q99 * 10 or min_val < q01 * 10:
                print(f"  {col}: OUTLIER DETECTED - Range: [{min_val:.2f}, {max_val:.2f}], Q99: {q99:.2f}")
            else:
                print(f"  {col}: OK - Range: [{min_val:.2f}, {max_val:.2f}]")
        else:
            print(f"  {col}: MISSING FROM TEST DATA")