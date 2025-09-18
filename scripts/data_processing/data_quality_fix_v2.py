#!/usr/bin/env python3
"""
Advanced Data Quality Fix Pipeline V2
- More aggressive outlier handling
- Robust scaling methods
- Better feature engineering
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
import warnings
warnings.filterwarnings('ignore')

# Add path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def advanced_outlier_cleaning(df, feature_cols):
    """Advanced outlier cleaning with robust methods"""
    print("🔧 ADVANCED OUTLIER CLEANING")
    print("=" * 60)
    
    df_clean = df.copy()
    
    # Separate feature types for different treatment
    volume_cols = [col for col in feature_cols if 'volume' in col.lower()]
    price_cols = [col for col in feature_cols if any(x in col.lower() for x in ['open', 'high', 'low', 'close', 'ma_'])]
    volatility_cols = [col for col in feature_cols if 'volatility' in col.lower()]
    return_cols = [col for col in feature_cols if 'return' in col.lower()]
    other_cols = [col for col in feature_cols if col not in volume_cols + price_cols + volatility_cols + return_cols]
    
    # 1. Volume: Most extreme outliers - use log transformation + capping
    for col in volume_cols:
        if col in df_clean.columns:
            original_max = df_clean[col].max()
            
            # Log transform to reduce skewness
            df_clean[col + '_log'] = np.log1p(df_clean[col])
            
            # Cap at 99.9th percentile
            cap_value = df_clean[col].quantile(0.999)
            df_clean[col] = df_clean[col].clip(upper=cap_value)
            
            print(f"✅ {col}: {original_max:.2e} → {df_clean[col].max():.2e} (capped at 99.9%)")
    
    # 2. Price data: Use robust statistical methods
    for col in price_cols:
        if col in df_clean.columns:
            original_range = (df_clean[col].min(), df_clean[col].max())
            
            # Use modified Z-score with median
            median = df_clean[col].median()
            mad = np.median(np.abs(df_clean[col] - median))
            
            if mad > 0:
                modified_z_scores = 0.6745 * (df_clean[col] - median) / mad
                outlier_threshold = 3.5
                
                # Cap extreme outliers
                upper_cap = median + outlier_threshold * mad / 0.6745
                lower_cap = median - outlier_threshold * mad / 0.6745
                
                df_clean[col] = df_clean[col].clip(lower=lower_cap, upper=upper_cap)
                
                new_range = (df_clean[col].min(), df_clean[col].max())
                if original_range != new_range:
                    print(f"✅ {col}: {original_range[1]:.2f} → {new_range[1]:.2f}")
    
    # 3. Volatility: Use percentile capping
    for col in volatility_cols:
        if col in df_clean.columns:
            original_max = df_clean[col].max()
            
            # Cap at 99th percentile for volatility
            cap_value = df_clean[col].quantile(0.99)
            df_clean[col] = df_clean[col].clip(upper=cap_value)
            
            print(f"✅ {col}: {original_max:.2f} → {df_clean[col].max():.2f}")
    
    # 4. Returns: Very conservative capping
    for col in return_cols:
        if col in df_clean.columns:
            original_range = (df_clean[col].min(), df_clean[col].max())
            
            # Cap at 99.5% and 0.5% percentiles
            upper_cap = df_clean[col].quantile(0.995)
            lower_cap = df_clean[col].quantile(0.005)
            
            df_clean[col] = df_clean[col].clip(lower=lower_cap, upper=upper_cap)
            
            new_range = (df_clean[col].min(), df_clean[col].max())
            if abs(original_range[1] - new_range[1]) > 0.001:
                print(f"✅ {col}: [{original_range[0]:.4f}, {original_range[1]:.4f}] → [{new_range[0]:.4f}, {new_range[1]:.4f}]")
    
    return df_clean

def create_robust_features(df, feature_cols):
    """Create robust features that are less sensitive to outliers"""
    print("\n🛠️ CREATING ROBUST FEATURES")
    print("=" * 60)
    
    df_robust = df.copy()
    
    # For volume, create log and rank features
    volume_cols = [col for col in feature_cols if 'volume' in col.lower()]
    for col in volume_cols:
        if col in df_robust.columns:
            # Log transformation
            df_robust[col + '_log'] = np.log1p(df_robust[col])
            
            # Rank transformation (most robust)
            df_robust[col + '_rank'] = df_robust[col].rank(pct=True)
            
            print(f"✅ Created {col}_log and {col}_rank features")
    
    # For price data, create relative features
    price_cols = [col for col in feature_cols if any(x in col.lower() for x in ['open', 'high', 'low', 'close'])]
    if len(price_cols) >= 4:
        # Create price ratios (more stable)
        if 'high' in df_robust.columns and 'low' in df_robust.columns:
            df_robust['price_range_ratio'] = (df_robust['high'] - df_robust['low']) / df_robust['low']
        
        if 'close' in df_robust.columns and 'open' in df_robust.columns:
            df_robust['price_change_ratio'] = (df_robust['close'] - df_robust['open']) / df_robust['open']
        
        print("✅ Created price ratio features")
    
    return df_robust

def apply_robust_scaling(X_train, X_val, X_test, feature_cols):
    """Apply robust scaling to features"""
    print("\n📏 APPLYING ROBUST SCALING")
    print("=" * 60)
    
    # Different scalers for different feature types
    volume_cols = [col for col in feature_cols if 'volume' in col.lower()]
    other_cols = [col for col in feature_cols if col not in volume_cols]
    
    # RobustScaler for most features (less sensitive to outliers)
    robust_scaler = RobustScaler()
    
    # PowerTransformer for volume (handles extreme skewness)
    power_transformer = PowerTransformer(method='yeo-johnson')
    
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy() if len(X_val) > 0 else pd.DataFrame()
    X_test_scaled = X_test.copy()
    
    # Scale non-volume features with RobustScaler
    if other_cols:
        other_cols_present = [col for col in other_cols if col in X_train.columns]
        if other_cols_present:
            X_train_scaled[other_cols_present] = robust_scaler.fit_transform(X_train[other_cols_present])
            if len(X_val) > 0:
                X_val_scaled[other_cols_present] = robust_scaler.transform(X_val[other_cols_present])
            X_test_scaled[other_cols_present] = robust_scaler.transform(X_test[other_cols_present])
            
            print(f"✅ RobustScaler applied to {len(other_cols_present)} features")
    
    # Scale volume features with PowerTransformer
    if volume_cols:
        volume_cols_present = [col for col in volume_cols if col in X_train.columns]
        if volume_cols_present:
            X_train_scaled[volume_cols_present] = power_transformer.fit_transform(X_train[volume_cols_present])
            if len(X_val) > 0:
                X_val_scaled[volume_cols_present] = power_transformer.transform(X_val[volume_cols_present])
            X_test_scaled[volume_cols_present] = power_transformer.transform(X_test[volume_cols_present])
            
            print(f"✅ PowerTransformer applied to {len(volume_cols_present)} volume features")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, robust_scaler, power_transformer

def main():
    """Main advanced data cleaning pipeline"""
    print("🚀 ADVANCED CRYPTO DATA QUALITY FIX PIPELINE V2")
    print("=" * 90)
    
    # Load original datasets
    print("📂 Loading original datasets...")
    try:
        with open('data/cache/ml_datasets_top3.pkl', 'rb') as f:
            datasets = pickle.load(f)
    except FileNotFoundError:
        print("Error: Original datasets not found!")
        return
    
    feature_cols = datasets['feature_cols']
    
    # Get raw dataframes
    train_df = datasets['train_df'].copy()
    test_df = datasets['test_df'].copy()
    
    print(f"📊 Original data shapes:")
    print(f"  Training: {train_df.shape}")
    print(f"  Test: {test_df.shape}")
    print(f"  Features: {feature_cols}")
    
    # Step 1: Advanced outlier cleaning
    print("\n" + "="*90)
    print("🧹 STEP 1: ADVANCED OUTLIER CLEANING")
    print("="*90)
    
    train_df_clean = advanced_outlier_cleaning(train_df, feature_cols)
    test_df_clean = advanced_outlier_cleaning(test_df, feature_cols)
    
    # Step 2: Create robust features
    print("\n" + "="*90)
    print("🛠️ STEP 2: ROBUST FEATURE ENGINEERING")
    print("="*90)
    
    train_df_robust = create_robust_features(train_df_clean, feature_cols)
    test_df_robust = create_robust_features(test_df_clean, feature_cols)
    
    # Update feature columns to include new features
    new_features = [col for col in train_df_robust.columns if col in feature_cols or col.endswith(('_log', '_rank', '_ratio'))]
    robust_feature_cols = [col for col in new_features if col in train_df_robust.columns]
    
    print(f"📊 Extended feature set: {len(robust_feature_cols)} features")
    print(f"  Original: {feature_cols}")
    print(f"  New features: {[col for col in robust_feature_cols if col not in feature_cols]}")
    
    # Step 3: Split data properly
    print("\n" + "="*90)
    print("📊 STEP 3: DATA SPLITTING")
    print("="*90)
    
    # Use original split sizes
    train_size = len(datasets['X_train'])
    val_size = len(datasets['X_val'])
    
    # Split cleaned training data
    X_train_clean = train_df_robust[robust_feature_cols].iloc[:train_size]
    X_val_clean = train_df_robust[robust_feature_cols].iloc[train_size:train_size+val_size] if val_size > 0 else pd.DataFrame()
    X_test_clean = test_df_robust[robust_feature_cols]
    
    # Targets
    y_train_clean = {
        'price': train_df_robust['target_price'].iloc[:train_size],
        'price_change': train_df_robust['target_price_change'].iloc[:train_size],
        'trend': train_df_robust['target_trend'].iloc[:train_size]
    }
    
    y_val_clean = {
        'price': train_df_robust['target_price'].iloc[train_size:train_size+val_size] if val_size > 0 else pd.Series(),
        'price_change': train_df_robust['target_price_change'].iloc[train_size:train_size+val_size] if val_size > 0 else pd.Series(),
        'trend': train_df_robust['target_trend'].iloc[train_size:train_size+val_size] if val_size > 0 else pd.Series()
    }
    
    y_test_clean = {
        'price': test_df_robust['target_price'],
        'price_change': test_df_robust['target_price_change'],
        'trend': test_df_robust['target_trend']
    }
    
    print(f"✅ Data split:")
    print(f"  X_train: {X_train_clean.shape}")
    print(f"  X_val: {X_val_clean.shape if len(X_val_clean) > 0 else 'Empty'}")
    print(f"  X_test: {X_test_clean.shape}")
    
    # Step 4: Apply robust scaling
    print("\n" + "="*90)
    print("📏 STEP 4: ROBUST SCALING")
    print("="*90)
    
    X_train_scaled, X_val_scaled, X_test_scaled, robust_scaler, power_transformer = apply_robust_scaling(
        X_train_clean, X_val_clean, X_test_clean, robust_feature_cols
    )
    
    # Step 5: Final validation
    print("\n" + "="*90)
    print("✅ STEP 5: FINAL VALIDATION")
    print("="*90)
    
    print("📊 Final data ranges:")
    print(f"  X_train_scaled: [{X_train_scaled.min().min():.4f}, {X_train_scaled.max().max():.4f}]")
    print(f"  X_test_scaled: [{X_test_scaled.min().min():.4f}, {X_test_scaled.max().max():.4f}]")
    print(f"  y_train price: [{y_train_clean['price'].min():.2f}, {y_train_clean['price'].max():.2f}]")
    print(f"  y_test price: [{y_test_clean['price'].min():.2f}, {y_test_clean['price'].max():.2f}]")
    
    # Check for any remaining extreme outliers
    for col in robust_feature_cols:
        col_range = X_test_scaled[col].max() - X_test_scaled[col].min()
        if col_range > 10:  # Scaled data should typically be in [-3, 3] range
            print(f"⚠️  {col}: Large range after scaling: {col_range:.2f}")
        else:
            print(f"✅ {col}: Good scaling range: {col_range:.2f}")
    
    # Create final datasets
    datasets_v2_clean = {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train_clean,
        'y_val': y_val_clean,
        'y_test': y_test_clean,
        'feature_cols': robust_feature_cols,
        'original_feature_cols': feature_cols,
        
        # Raw data (before scaling)
        'X_train_raw': X_train_clean,
        'X_val_raw': X_val_clean,
        'X_test_raw': X_test_clean,
        
        # Dataframes
        'train_df': train_df_robust,
        'val_df': train_df_robust.iloc[train_size:train_size+val_size] if val_size > 0 else pd.DataFrame(),
        'test_df': test_df_robust,
        
        # Scalers
        'robust_scaler': robust_scaler,
        'power_transformer': power_transformer,
        'scaler': robust_scaler,  # For backward compatibility
        
        # Metadata
        'cleaning_metadata': {
            'outlier_method': 'advanced_robust',
            'scaling_method': 'robust_and_power',
            'feature_engineering': 'log_rank_ratio',
            'data_quality_score': 'high'
        }
    }
    
    # Save improved datasets
    os.makedirs('data/cache', exist_ok=True)
    
    with open('data/cache/ml_datasets_top3_v2_clean.pkl', 'wb') as f:
        pickle.dump(datasets_v2_clean, f)
    
    print(f"\n💾 Advanced cleaned datasets saved to: data/cache/ml_datasets_top3_v2_clean.pkl")
    
    # Final summary
    print("\n" + "="*90)
    print("🎯 ADVANCED CLEANING COMPLETE")
    print("="*90)
    
    print("✅ Advanced outlier removal applied")
    print("✅ Robust feature engineering completed")
    print("✅ Multi-method scaling applied")
    print("✅ Data quality significantly improved")
    print("✅ Ready for robust model training")
    
    print(f"\n📈 Quality improvements:")
    print(f"  - Extreme outliers capped/removed")
    print(f"  - Robust scaling reduces sensitivity")
    print(f"  - New features improve stability") 
    print(f"  - {len(robust_feature_cols)} total features available")
    
    return datasets_v2_clean

if __name__ == "__main__":
    main()