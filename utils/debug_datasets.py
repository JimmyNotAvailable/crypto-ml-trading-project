#!/usr/bin/env python3

import pickle
import sys
import os
import numpy as np
sys.path.append('app')

# Load datasets
print("Loading datasets...")
with open('data/cache/ml_datasets_top3.pkl', 'rb') as f:
    datasets = pickle.load(f)

print("=== DATASETS CONTENT ANALYSIS ===")
print(f"Keys in datasets: {list(datasets.keys())}")

# Compare direct X_test/y_test vs test_df data
X_test_direct = datasets['X_test']
y_test_direct = datasets['y_test']['price']
feature_cols = datasets['feature_cols']

test_df = datasets['test_df']
X_test_from_df = test_df[feature_cols].fillna(0)
y_test_from_df = test_df['target_price']

print(f"\n=== SHAPES ===")
print(f"X_test_direct: {X_test_direct.shape}")
print(f"y_test_direct: {y_test_direct.shape}")
print(f"X_test_from_df: {X_test_from_df.shape}")
print(f"y_test_from_df: {y_test_from_df.shape}")

print(f"\n=== DATA TYPES ===")
print(f"X_test_direct type: {type(X_test_direct)}")
print(f"y_test_direct type: {type(y_test_direct)}")
print(f"X_test_from_df type: {type(X_test_from_df)}")
print(f"y_test_from_df type: {type(y_test_from_df)}")

print(f"\n=== X DATA COMPARISON ===")
print(f"X_test_direct range: [{np.min(X_test_direct):.4f}, {np.max(X_test_direct):.4f}]")
print(f"X_test_from_df range: [{X_test_from_df.min().min():.4f}, {X_test_from_df.max().max():.4f}]")

print(f"\n=== Y DATA COMPARISON ===")
print(f"y_test_direct range: [{np.min(y_test_direct):.4f}, {np.max(y_test_direct):.4f}]")
print(f"y_test_from_df range: [{y_test_from_df.min():.4f}, {y_test_from_df.max():.4f}]")

# Check if they're equal
print(f"\n=== EQUALITY CHECKS ===")
if hasattr(X_test_direct, 'values'):
    x_equal = np.allclose(X_test_direct.values, X_test_from_df.values, equal_nan=True)
else:
    x_equal = np.allclose(X_test_direct, X_test_from_df.values, equal_nan=True)

y_equal = np.allclose(y_test_direct, y_test_from_df, equal_nan=True)

print(f"X data equal: {x_equal}")
print(f"Y data equal: {y_equal}")

# Sample comparison
print(f"\n=== SAMPLE DATA (first 5 rows) ===")
print("X_test_direct[0:5]:")
print(X_test_direct[:5])
print("\nX_test_from_df[0:5]:")
print(X_test_from_df.iloc[:5].values)

print(f"\ny_test_direct[0:5]: {y_test_direct[:5]}")
print(f"y_test_from_df[0:5]: {y_test_from_df.iloc[:5].values}")

# Check feature order in X_test_direct 
if hasattr(X_test_direct, 'columns'):
    print(f"\n=== FEATURE COLUMNS ===")
    print(f"X_test_direct columns: {list(X_test_direct.columns)}")
    print(f"X_test_from_df columns: {list(X_test_from_df.columns)}")
    print(f"datasets feature_cols: {feature_cols}")