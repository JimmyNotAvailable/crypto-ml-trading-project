# core.py
# Core ML utilities and shared functions

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from abc import ABC, abstractmethod
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ModelInterface(ABC):
    """Abstract base class for all ML models"""
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Train the model and return metrics"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """Save trained model to file"""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """Load trained model from file"""
        pass

class DataValidator:
    """Data validation utilities for ML pipeline"""
    
    @staticmethod
    def validate_features(df: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate that DataFrame has required columns"""
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        return True
    
    @staticmethod
    def validate_data_types(df: pd.DataFrame, expected_types: Dict[str, str]) -> bool:
        """Validate data types of DataFrame columns"""
        for col, expected_type in expected_types.items():
            if col in df.columns:
                if expected_type == 'numeric':
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        raise ValueError(f"Column {col} should be numeric, got {df[col].dtype}")
                elif expected_type == 'datetime':
                    if not pd.api.types.is_datetime64_any_dtype(df[col]):
                        raise ValueError(f"Column {col} should be datetime, got {df[col].dtype}")
        return True
    
    @staticmethod
    def check_missing_values(df: pd.DataFrame, max_missing_ratio: float = 0.1) -> Dict[str, float]:
        """Check for missing values and return ratios"""
        missing_ratios = df.isnull().sum() / len(df)
        problematic_cols = missing_ratios[missing_ratios > max_missing_ratio]
        
        if not problematic_cols.empty:
            warnings.warn(f"High missing value ratios detected: {problematic_cols.to_dict()}")
        
        return missing_ratios.to_dict()
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = 3.0) -> Dict[str, List[int]]:
        """Detect outliers in numeric columns"""
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_indices = df[z_scores > threshold].index.tolist()
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            outliers[col] = outlier_indices
        
        return outliers

class ModelEvaluator:
    """Model evaluation utilities"""
    
    @staticmethod
    def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Standard regression metrics"""
        return {
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'r2': float(r2_score(y_true, y_pred)),
            'mape': float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
            'max_error': float(np.max(np.abs(y_true - y_pred)))
        }
    
    @staticmethod
    def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Standard classification metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }

class ModelPersistence:
    """Model saving and loading utilities"""
    
    @staticmethod
    def save_model_with_metadata(model: Any, metadata: Dict[str, Any], 
                                model_path: str, metadata_path: str) -> None:
        """Save model and its metadata"""
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        # Save model
        joblib.dump(model, model_path)
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    @staticmethod
    def load_model_with_metadata(model_path: str, metadata_path: str) -> Tuple[Any, Dict[str, Any]]:
        """Load model and its metadata"""
        # Load model
        model = joblib.load(model_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return model, metadata

class FeatureEngineering:
    """Feature engineering utilities"""
    
    @staticmethod
    def create_lag_features(df: pd.DataFrame, columns: List[str], 
                          lags: List[int]) -> pd.DataFrame:
        """Create lagged features"""
        df_with_lags = df.copy()
        
        for col in columns:
            for lag in lags:
                df_with_lags[f"{col}_lag_{lag}"] = df[col].shift(lag)
        
        return df_with_lags
    
    @staticmethod
    def create_rolling_features(df: pd.DataFrame, columns: List[str],
                              windows: List[int], agg_funcs: List[str] = ['mean', 'std']) -> pd.DataFrame:
        """Create rolling window features"""
        df_with_rolling = df.copy()
        
        for col in columns:
            for window in windows:
                for agg_func in agg_funcs:
                    feature_name = f"{col}_rolling_{window}_{agg_func}"
                    df_with_rolling[feature_name] = df[col].rolling(window=window).agg(agg_func)
        
        return df_with_rolling
    
    @staticmethod
    def create_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Create technical analysis indicators"""
        df_tech = df.copy()
        
        # RSI
        if 'close' in df.columns:
            delta = df['close'].diff()
            # Fix the comparison by converting to numeric first
            delta_numeric = pd.to_numeric(delta, errors='coerce')
            gain = (delta_numeric.where(delta_numeric > 0, 0)).rolling(window=14).mean()
            loss = (-delta_numeric.where(delta_numeric < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_tech['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        if 'close' in df.columns:
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df_tech['macd'] = exp1 - exp2
            df_tech['macd_signal'] = df_tech['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        if 'close' in df.columns:
            df_tech['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df_tech['bb_upper'] = df_tech['bb_middle'] + (bb_std * 2)
            df_tech['bb_lower'] = df_tech['bb_middle'] - (bb_std * 2)
            df_tech['bb_width'] = df_tech['bb_upper'] - df_tech['bb_lower']
        
        return df_tech

class DataSplitter:
    """Data splitting utilities for time series"""
    
    @staticmethod
    def time_series_split(df: pd.DataFrame, train_ratio: float = 0.7, 
                         val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split time series data maintaining temporal order"""
        n = len(df)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        test_df = df.iloc[train_size + val_size:]
        
        return train_df, val_df, test_df
    
    @staticmethod
    def stratified_time_split(df: pd.DataFrame, target_col: str,
                            train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data maintaining class distribution over time"""
        # Implementation for stratified time series split
        # This is more complex and would require custom logic
        # For now, fall back to regular time series split
        return DataSplitter.time_series_split(df, train_ratio, val_ratio)

class ConfigManager:
    """Configuration management utilities"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> None:
        """Save configuration to JSON file"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    @staticmethod
    def get_project_root() -> Path:
        """Get project root directory"""
        current_file = Path(__file__).resolve()
        # Go up from app/ml/core.py to project root
        return current_file.parent.parent.parent

class CryptoUtils:
    """Crypto-specific utilities"""
    
    @staticmethod
    def calculate_volatility(prices: pd.Series, window: int = 24) -> pd.Series:
        """Calculate rolling volatility"""
        returns = prices.pct_change()
        return returns.rolling(window=window).std() * np.sqrt(window)
    
    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """Calculate price returns"""
        return prices.pct_change()
    
    @staticmethod
    def detect_regime_change(prices: pd.Series, window: int = 50) -> pd.Series:
        """Detect market regime changes"""
        ma_short = prices.rolling(window=window//2).mean()
        ma_long = prices.rolling(window=window).mean()
        
        # Bull market when short MA > long MA
        regime = (ma_short > ma_long).astype(int)
        return regime
    
    @staticmethod
    def calculate_drawdown(prices: pd.Series) -> pd.Series:
        """Calculate rolling maximum drawdown"""
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak
        return drawdown

# Global utility functions
def setup_logging(log_level: str = 'INFO') -> None:
    """Setup logging configuration"""
    import logging
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ml_pipeline.log'),
            logging.StreamHandler()
        ]
    )

def get_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
    """Get memory usage information for DataFrame"""
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum()
    
    return {
        'total_memory_mb': f"{total_memory / 1024**2:.2f} MB",
        'memory_per_column': {col: f"{mem / 1024**2:.2f} MB" 
                            for col, mem in memory_usage.items()},
        'dataframe_size': f"{df.shape[0]} rows x {df.shape[1]} columns"
    }

def validate_model_performance(metrics: Dict[str, float], 
                             min_r2: float = 0.8, max_mae_ratio: float = 0.1) -> bool:
    """Validate if model performance meets minimum requirements"""
    if 'r2' in metrics and metrics['r2'] < min_r2:
        warnings.warn(f"R² score {metrics['r2']:.4f} below minimum {min_r2}")
        return False
    
    if 'mae' in metrics and 'target_mean' in metrics:
        mae_ratio = metrics['mae'] / metrics['target_mean']
        if mae_ratio > max_mae_ratio:
            warnings.warn(f"MAE ratio {mae_ratio:.4f} above maximum {max_mae_ratio}")
            return False
    
    return True
