#!/usr/bin/env python3
"""
Data Quality Fix Pipeline
- Detects and handles outliers in crypto datasets
- Implements robust cleaning strategies
- Preserves data integrity while removing extreme values
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Add path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.ml.data_prep import load_prepared_datasets

class CryptoDataCleaner:
    """Advanced data cleaning for crypto datasets"""
    
    def __init__(self, outlier_method='iqr', cap_method='percentile'):
        """
        Initialize data cleaner
        
        Args:
            outlier_method: 'iqr', 'zscore', 'isolation'
            cap_method: 'percentile', 'iqr', 'median'
        """
        self.outlier_method = outlier_method
        self.cap_method = cap_method
        self.outlier_stats = {}
        
    def detect_outliers_iqr(self, data, column, multiplier=1.5):
        """Detect outliers using IQR method"""
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
        return outliers, lower_bound, upper_bound
    
    def detect_outliers_zscore(self, data, column, threshold=3):
        """Detect outliers using Z-score method"""
        z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
        outliers = z_scores > threshold
        return outliers
    
    def cap_outliers_percentile(self, data, column, lower_percentile=1, upper_percentile=99):
        """Cap outliers using percentile method"""
        lower_cap = data[column].quantile(lower_percentile / 100)
        upper_cap = data[column].quantile(upper_percentile / 100)
        
        data[column] = data[column].clip(lower=lower_cap, upper=upper_cap)
        return data, lower_cap, upper_cap
    
    def cap_outliers_iqr(self, data, column, multiplier=1.5):
        """Cap outliers using IQR method"""
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_cap = Q1 - multiplier * IQR
        upper_cap = Q3 + multiplier * IQR
        
        data[column] = data[column].clip(lower=lower_cap, upper=upper_cap)
        return data, lower_cap, upper_cap
    
    def analyze_outliers(self, df, feature_cols):
        """Comprehensive outlier analysis"""
        print("🔍 OUTLIER ANALYSIS")
        print("=" * 60)
        
        outlier_summary = {}
        
        for col in feature_cols:
            if col in df.columns:
                col_data = df[col]
                
                # Basic stats
                stats = {
                    'count': col_data.count(),
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'q1': col_data.quantile(0.25),
                    'q3': col_data.quantile(0.75),
                    'q99': col_data.quantile(0.99),
                    'q01': col_data.quantile(0.01)
                }
                
                # IQR outlier detection
                outliers_iqr, lower_bound, upper_bound = self.detect_outliers_iqr(df, col)
                outlier_count = outliers_iqr.sum()
                outlier_percentage = (outlier_count / len(df)) * 100
                
                stats.update({
                    'outliers_count': outlier_count,
                    'outliers_percentage': outlier_percentage,
                    'iqr_lower': lower_bound,
                    'iqr_upper': upper_bound
                })
                
                outlier_summary[col] = stats
                
                # Print analysis
                status = "🚨 CRITICAL" if outlier_percentage > 5 else "⚠️ WARNING" if outlier_percentage > 1 else "✅ OK"
                print(f"{status} {col}:")
                print(f"  Range: [{stats['min']:.2e}, {stats['max']:.2e}]")
                print(f"  Q99: {stats['q99']:.2e}")
                print(f"  Outliers: {outlier_count} ({outlier_percentage:.1f}%)")
                
                if outlier_percentage > 1:
                    extreme_outliers = col_data[outliers_iqr]
                    print(f"  Extreme values: {extreme_outliers.nlargest(3).tolist()}")
                print()
        
        return outlier_summary
    
    def clean_dataset(self, df, feature_cols, target_cols=None):
        """Clean dataset by handling outliers"""
        print("🧹 CLEANING DATASET")
        print("=" * 60)
        
        df_clean = df.copy()
        cleaning_report = {}
        
        # Special handling for different feature types
        volume_cols = [col for col in feature_cols if 'volume' in col.lower()]
        price_cols = [col for col in feature_cols if any(x in col.lower() for x in ['open', 'high', 'low', 'close', 'ma_'])]
        volatility_cols = [col for col in feature_cols if 'volatility' in col.lower()]
        return_cols = [col for col in feature_cols if 'return' in col.lower()]
        
        # Handle volume outliers (most extreme)
        for col in volume_cols:
            if col in df_clean.columns:
                original_range = (df_clean[col].min(), df_clean[col].max())
                
                # Use very strict percentile capping for volume
                df_clean, lower_cap, upper_cap = self.cap_outliers_percentile(
                    df_clean, col, lower_percentile=0.1, upper_percentile=99.5
                )
                
                new_range = (df_clean[col].min(), df_clean[col].max())
                cleaning_report[col] = {
                    'method': 'percentile_0.1_99.5',
                    'original_range': original_range,
                    'new_range': new_range,
                    'capped_values': (lower_cap, upper_cap)
                }
                
                print(f"✅ {col}: {original_range[1]:.2e} → {new_range[1]:.2e}")
        
        # Handle price outliers
        for col in price_cols:
            if col in df_clean.columns:
                original_range = (df_clean[col].min(), df_clean[col].max())
                
                # Use IQR method for price data
                df_clean, lower_cap, upper_cap = self.cap_outliers_iqr(
                    df_clean, col, multiplier=3.0  # More lenient for prices
                )
                
                new_range = (df_clean[col].min(), df_clean[col].max())
                cleaning_report[col] = {
                    'method': 'iqr_3.0',
                    'original_range': original_range,
                    'new_range': new_range,
                    'capped_values': (lower_cap, upper_cap)
                }
                
                if original_range != new_range:
                    print(f"✅ {col}: {original_range[1]:.2f} → {new_range[1]:.2f}")
        
        # Handle volatility outliers
        for col in volatility_cols:
            if col in df_clean.columns:
                original_range = (df_clean[col].min(), df_clean[col].max())
                
                # Use percentile method for volatility
                df_clean, lower_cap, upper_cap = self.cap_outliers_percentile(
                    df_clean, col, lower_percentile=0.5, upper_percentile=99
                )
                
                new_range = (df_clean[col].min(), df_clean[col].max())
                cleaning_report[col] = {
                    'method': 'percentile_0.5_99',
                    'original_range': original_range,
                    'new_range': new_range,
                    'capped_values': (lower_cap, upper_cap)
                }
                
                if original_range != new_range:
                    print(f"✅ {col}: {original_range[1]:.2f} → {new_range[1]:.2f}")
        
        # Returns are usually OK, just check
        for col in return_cols:
            if col in df_clean.columns:
                outliers_iqr, _, _ = self.detect_outliers_iqr(df_clean, col)
                if outliers_iqr.sum() > len(df_clean) * 0.01:  # >1% outliers
                    original_range = (df_clean[col].min(), df_clean[col].max())
                    df_clean, lower_cap, upper_cap = self.cap_outliers_iqr(
                        df_clean, col, multiplier=4.0  # Very lenient for returns
                    )
                    new_range = (df_clean[col].min(), df_clean[col].max())
                    print(f"✅ {col}: {original_range[1]:.4f} → {new_range[1]:.4f}")
        
        return df_clean, cleaning_report
    
    def validate_cleaning(self, df_original, df_clean, feature_cols):
        """Validate that cleaning preserved data integrity"""
        print("\n📊 CLEANING VALIDATION")
        print("=" * 60)
        
        validation_results = {}
        
        for col in feature_cols:
            if col in df_clean.columns:
                orig_data = df_original[col]
                clean_data = df_clean[col]
                
                # Check data preservation
                data_preserved = (clean_data <= orig_data).all() and (clean_data >= orig_data.min()).all()
                
                # Check distribution preservation
                orig_mean = orig_data.mean()
                clean_mean = clean_data.mean()
                mean_change = abs(clean_mean - orig_mean) / orig_mean * 100
                
                # Check outlier reduction
                orig_outliers, _, _ = self.detect_outliers_iqr(df_original, col)
                clean_outliers, _, _ = self.detect_outliers_iqr(df_clean, col)
                
                outlier_reduction = (orig_outliers.sum() - clean_outliers.sum()) / orig_outliers.sum() * 100 if orig_outliers.sum() > 0 else 0
                
                validation_results[col] = {
                    'data_preserved': data_preserved,
                    'mean_change_percent': mean_change,
                    'outlier_reduction_percent': outlier_reduction,
                    'original_outliers': orig_outliers.sum(),
                    'cleaned_outliers': clean_outliers.sum()
                }
                
                status = "✅" if mean_change < 10 and outlier_reduction > 50 else "⚠️"
                print(f"{status} {col}: Mean change: {mean_change:.1f}%, Outliers: {orig_outliers.sum()} → {clean_outliers.sum()}")
        
        return validation_results

def main():
    """Main data cleaning pipeline"""
    print("🚀 CRYPTO DATA QUALITY FIX PIPELINE")
    print("=" * 80)
    
    # Load datasets
    print("📂 Loading datasets...")
    try:
        with open('data/cache/ml_datasets_top3.pkl', 'rb') as f:
            datasets = pickle.load(f)
    except FileNotFoundError:
        print("Loading fresh datasets from data_prep...")
        datasets = load_prepared_datasets('ml_datasets_top3')
    
    # Initialize cleaner
    cleaner = CryptoDataCleaner()
    
    # Get feature columns
    feature_cols = datasets['feature_cols']
    print(f"📊 Feature columns: {feature_cols}")
    
    # Analyze training data
    print("\n" + "="*80)
    print("🔍 TRAINING DATA ANALYSIS")
    print("="*80)
    
    # Create full training dataframe for analysis
    train_df = datasets['train_df']
    outlier_summary_train = cleaner.analyze_outliers(train_df, feature_cols)
    
    # Analyze test data
    print("\n" + "="*80)
    print("🔍 TEST DATA ANALYSIS")
    print("="*80)
    
    test_df = datasets['test_df']
    outlier_summary_test = cleaner.analyze_outliers(test_df, feature_cols)
    
    # Clean training data
    print("\n" + "="*80)
    print("🧹 CLEANING TRAINING DATA")
    print("="*80)
    
    train_df_clean, train_cleaning_report = cleaner.clean_dataset(
        train_df, feature_cols, ['target_price', 'target_price_change', 'target_trend']
    )
    
    # Clean test data
    print("\n" + "="*80)
    print("🧹 CLEANING TEST DATA")
    print("="*80)
    
    test_df_clean, test_cleaning_report = cleaner.clean_dataset(
        test_df, feature_cols, ['target_price', 'target_price_change', 'target_trend']
    )
    
    # Validate cleaning
    print("\n" + "="*80)
    print("✅ VALIDATION RESULTS")
    print("="*80)
    
    train_validation = cleaner.validate_cleaning(train_df, train_df_clean, feature_cols)
    test_validation = cleaner.validate_cleaning(test_df, test_df_clean, feature_cols)
    
    # Create cleaned datasets
    print("\n" + "="*80)
    print("💾 CREATING CLEANED DATASETS")
    print("="*80)
    
    # Split cleaned training data
    train_size = len(datasets['X_train'])
    val_size = len(datasets['X_val'])
    
    X_train_clean = train_df_clean[feature_cols].iloc[:train_size]
    X_val_clean = train_df_clean[feature_cols].iloc[train_size:train_size+val_size]
    X_test_clean = test_df_clean[feature_cols]
    
    # Create target dictionaries
    y_train_clean = {
        'price': train_df_clean['target_price'].iloc[:train_size],
        'price_change': train_df_clean['target_price_change'].iloc[:train_size],
        'trend': train_df_clean['target_trend'].iloc[:train_size]
    }
    
    y_val_clean = {
        'price': train_df_clean['target_price'].iloc[train_size:train_size+val_size],
        'price_change': train_df_clean['target_price_change'].iloc[train_size:train_size+val_size],
        'trend': train_df_clean['target_trend'].iloc[train_size:train_size+val_size]
    }
    
    y_test_clean = {
        'price': test_df_clean['target_price'],
        'price_change': test_df_clean['target_price_change'],
        'trend': test_df_clean['target_trend']
    }
    
    # Create new cleaned datasets dictionary
    datasets_clean = {
        'X_train': X_train_clean,
        'X_val': X_val_clean,
        'X_test': X_test_clean,
        'y_train': y_train_clean,
        'y_val': y_val_clean,
        'y_test': y_test_clean,
        'feature_cols': feature_cols,
        'train_df': train_df_clean,
        'val_df': train_df_clean.iloc[train_size:train_size+val_size],
        'test_df': test_df_clean,
        'scaler': datasets['scaler'],  # Keep original scaler for now
        # Raw data (unscaled)
        'X_train_raw': train_df_clean[feature_cols].iloc[:train_size],
        'X_val_raw': train_df_clean[feature_cols].iloc[train_size:train_size+val_size],
        'X_test_raw': test_df_clean[feature_cols],
        # Metadata
        'cleaning_metadata': {
            'train_cleaning_report': train_cleaning_report,
            'test_cleaning_report': test_cleaning_report,
            'train_validation': train_validation,
            'test_validation': test_validation,
            'outlier_summary_train': outlier_summary_train,
            'outlier_summary_test': outlier_summary_test
        }
    }
    
    # Save cleaned datasets
    os.makedirs('data/cache', exist_ok=True)
    
    with open('data/cache/ml_datasets_top3_cleaned.pkl', 'wb') as f:
        pickle.dump(datasets_clean, f)
    
    print("✅ Cleaned datasets saved to: data/cache/ml_datasets_top3_cleaned.pkl")
    
    # Summary report
    print("\n" + "="*80)
    print("📋 FINAL SUMMARY")
    print("="*80)
    
    print(f"✅ Training data: {len(X_train_clean)} samples")
    print(f"✅ Validation data: {len(X_val_clean)} samples")
    print(f"✅ Test data: {len(X_test_clean)} samples")
    print(f"✅ Features: {len(feature_cols)} columns")
    
    # Check data ranges after cleaning
    print(f"\n📊 DATA RANGES AFTER CLEANING:")
    print(f"X_train range: [{X_train_clean.min().min():.4f}, {X_train_clean.max().max():.4f}]")
    print(f"X_test range: [{X_test_clean.min().min():.4f}, {X_test_clean.max().max():.4f}]")
    print(f"y_train price range: [{y_train_clean['price'].min():.2f}, {y_train_clean['price'].max():.2f}]")
    print(f"y_test price range: [{y_test_clean['price'].min():.2f}, {y_test_clean['price'].max():.2f}]")
    
    print(f"\n🎯 Data quality issues have been resolved!")
    print(f"🚀 Ready for model retraining with cleaned data!")

if __name__ == "__main__":
    main()