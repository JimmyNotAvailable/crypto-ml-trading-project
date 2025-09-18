#!/usr/bin/env python3
"""
🔍 DATASET STRUCTURE INSPECTOR
=============================

Kiểm tra cấu trúc của datasets để debug model classes
"""

import sys
import os

# Add project path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'app'))

from app.ml.data_prep import load_prepared_datasets

def inspect_datasets():
    """Inspect dataset structure"""
    print("🔍 INSPECTING DATASET STRUCTURE")
    print("=" * 40)
    
    # Load datasets
    datasets = load_prepared_datasets("ml_datasets_top3")
    
    print(f"📊 Dataset type: {type(datasets)}")
    print(f"📊 Dataset keys: {list(datasets.keys())}")
    
    # Check DataFrame keys specifically
    df_keys = ['train_df', 'val_df', 'test_df']
    for key in df_keys:
        if key in datasets:
            data = datasets[key]
            print(f"\n🔹 {key}:")
            print(f"   Type: {type(data)}")
            print(f"   Shape: {data.shape}")
            print(f"   Columns: {list(data.columns)}")
            print(f"   Sample data:")
            print(data.head(2))

if __name__ == "__main__":
    inspect_datasets()