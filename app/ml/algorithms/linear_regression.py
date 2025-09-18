"""
📈 LINEAR REGRESSION MODEL
========================

Enterprise-grade Linear Regression implementation for crypto price prediction:
- ✅ Multiple target types (price, price_change)
- ✅ Feature engineering & preprocessing
- ✅ Cross-validation & hyperparameter tuning
- ✅ Comprehensive evaluation metrics
- ✅ Model versioning & persistence
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from .base import BaseModel

class LinearRegressionModel(BaseModel):
    """
    📈 Linear Regression Model for Crypto Price Prediction
    
    Features:
    - Price & price change prediction
    - Feature engineering
    - Cross-validation
    - Model persistence
    """
    
    def __init__(self, target_type: str = 'price', normalize_features: bool = True):
        """
        Initialize Linear Regression model
        
        Args:
            target_type: 'price' or 'price_change'
            normalize_features: Whether to normalize input features
        """
        super().__init__(
            model_name=f"linreg_{target_type}",
            model_type="regression"
        )
        
        self.target_type = target_type
        self.normalize_features = normalize_features
        self.scaler = StandardScaler() if normalize_features else None
        self.feature_columns = None
        
        # Update metadata
        self.metadata.update({
            'target_type': target_type,
            'normalize_features': normalize_features,
            'algorithm': 'LinearRegression'
        })
    
    def _prepare_features(self, datasets: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        🔧 Prepare features and targets for training
        
        Args:
            datasets: Dict containing train/test data
            
        Returns:
            Tuple of (features, targets)
        """
        print(f"🔧 Preparing features for {self.target_type} prediction...")
        
        # Get training data
        train_data = datasets['train'].copy()
        
        # Define feature columns (exclude target and non-numeric)
        exclude_cols = ['date', 'symbol', 'target_price', 'target_price_change', 'target_trend']
        feature_cols = [col for col in train_data.columns if col not in exclude_cols]
        
        # Ensure we have features
        if not feature_cols:
            raise ValueError("❌ No feature columns found for training")
        
        self.feature_columns = feature_cols
        X = train_data[feature_cols].copy()
        
        # Prepare target based on target_type
        if self.target_type == 'price':
            y = train_data['target_price'].copy()
        elif self.target_type == 'price_change':
            y = train_data['target_price_change'].copy()
        else:
            raise ValueError(f"❌ Invalid target_type: {self.target_type}")
        
        # Remove rows with NaN values
        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_idx]
        y = y[valid_idx]
        
        print(f"✅ Features prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"📊 Target: {self.target_type} (mean: {y.mean():.4f}, std: {y.std():.4f})")
        
        return X, y
    
    def train(self, datasets: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, Any]:
        """
        🎯 Train Linear Regression model
        
        Args:
            datasets: Dict containing 'train' and 'test' dataframes
            **kwargs: Additional training parameters
            
        Returns:
            Dict with training metrics and results
        """
        print(f"🎯 Training Linear Regression for {self.target_type}...")
        
        # Prepare data
        X_train, y_train = self._prepare_features(datasets)
        
        # Normalize features if requested
        if self.normalize_features and self.scaler:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        else:
            X_train_scaled = X_train
        
        # Initialize and train model
        self.model = LinearRegression()
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions on training data
        y_pred_train = self.model.predict(X_train_scaled)
        
        # Calculate training metrics
        train_metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'train_r2': r2_score(y_train, y_pred_train)
        }
        
        # Cross-validation
        if len(X_train) > 100:  # Only if we have enough data
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train, 
                cv=TimeSeriesSplit(n_splits=5), 
                scoring='r2'
            )
            train_metrics['cv_r2_mean'] = cv_scores.mean()
            train_metrics['cv_r2_std'] = cv_scores.std()
        
        # Test on validation data if available
        if 'test' in datasets:
            test_metrics = self._evaluate_on_test(datasets['test'])
            train_metrics.update(test_metrics)
        
        # Update model state
        self.is_trained = True
        self.training_history = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'target_type': self.target_type,
            'n_features': len(self.feature_columns),
            'n_samples': len(X_train),
            'metrics': train_metrics
        }
        
        # Update metadata
        self.metadata.update({
            'training_samples': len(X_train),
            'feature_count': len(self.feature_columns),
            'last_trained': pd.Timestamp.now().isoformat()
        })
        
        # Print results
        print(f"✅ Training completed!")
        print(f"📊 R² Score: {train_metrics['train_r2']:.4f}")
        print(f"📊 RMSE: {train_metrics['train_rmse']:.4f}")
        if 'test_r2' in train_metrics:
            print(f"📊 Test R²: {train_metrics['test_r2']:.4f}")
        
        return train_metrics
    
    def _evaluate_on_test(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model on test data"""
        # Prepare test features
        X_test = test_data[self.feature_columns].copy()
        
        if self.target_type == 'price':
            y_test = test_data['target_price'].copy()
        else:
            y_test = test_data['target_price_change'].copy()
        
        # Remove NaN values
        valid_idx = ~(X_test.isnull().any(axis=1) | y_test.isnull())
        X_test = X_test[valid_idx]
        y_test = y_test[valid_idx]
        
        # Scale if needed
        if self.normalize_features and self.scaler:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        else:
            X_test_scaled = X_test
        
        # Predict
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        return {
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_r2': r2_score(y_test, y_pred_test)
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        🔮 Make predictions
        
        Args:
            X: Feature dataframe
            
        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("❌ Model must be trained before making predictions")
        
        # Ensure we have the right features
        if self.feature_columns:
            X = X[self.feature_columns].copy()
        
        # Scale if needed
        if self.normalize_features and self.scaler:
            X_scaled = self.scaler.transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        📊 Evaluate model performance
        
        Args:
            X: Test features
            y: Test targets
            
        Returns:
            Dict with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("❌ Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        return {
            'mae': mean_absolute_error(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred)
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        📊 Get feature importance (coefficients)
        
        Returns:
            DataFrame with feature names and coefficients
        """
        if not self.is_trained or not self.feature_columns:
            raise ValueError("❌ Model must be trained to get feature importance")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'coefficient': self.model.coef_,
            'abs_coefficient': np.abs(self.model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        return importance_df
    
    def plot_predictions(self, X: pd.DataFrame, y: pd.Series, title: str = None) -> plt.Figure:
        """
        📊 Plot predictions vs actual values
        
        Args:
            X: Features
            y: Actual values
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        if not self.is_trained:
            raise ValueError("❌ Model must be trained to plot predictions")
        
        y_pred = self.predict(X)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        ax1.scatter(y, y_pred, alpha=0.6)
        ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.set_title(f'Predictions vs Actual\\nR² = {r2_score(y, y_pred):.4f}')
        
        # Residuals plot
        residuals = y - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        
        if title:
            fig.suptitle(title, fontsize=14)
        else:
            fig.suptitle(f'Linear Regression - {self.target_type}', fontsize=14)
        
        plt.tight_layout()
        return fig