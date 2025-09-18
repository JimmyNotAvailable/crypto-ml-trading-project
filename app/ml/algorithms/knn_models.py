"""
🎯 KNN MODELS
============

Enterprise-grade KNN implementation for crypto analysis:
- ✅ KNNClassifier for trend prediction
- ✅ KNNRegressor for price prediction  
- ✅ Automatic hyperparameter tuning
- ✅ Cross-validation & evaluation
- ✅ Feature scaling & preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                           mean_absolute_error, mean_squared_error, r2_score)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

from .base import BaseModel

class KNNClassifier(BaseModel):
    """
    🎯 KNN Classifier for Crypto Trend Prediction
    
    Features:
    - Trend classification (up/down/stable)
    - Hyperparameter tuning
    - Cross-validation
    - Feature importance analysis
    """
    
    def __init__(self, n_neighbors: int = 5, auto_tune: bool = True):
        """
        Initialize KNN Classifier
        
        Args:
            n_neighbors: Number of neighbors (if auto_tune=False)
            auto_tune: Whether to automatically tune hyperparameters
        """
        super().__init__(
            model_name="knn_classifier",
            model_type="classification"
        )
        
        self.n_neighbors = n_neighbors
        self.auto_tune = auto_tune
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.classes_ = None
        
        # Update metadata
        self.metadata.update({
            'n_neighbors': n_neighbors,
            'auto_tune': auto_tune,
            'algorithm': 'KNeighborsClassifier'
        })
    
    def _prepare_features(self, datasets: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        🔧 Prepare features and targets for classification
        
        Args:
            datasets: Dict containing train/test data
            
        Returns:
            Tuple of (features, encoded_targets)
        """
        print("🔧 Preparing features for trend classification...")
        
        # Get training data
        train_data = datasets['train'].copy()
        
        # Define feature columns
        exclude_cols = ['date', 'symbol', 'target_price', 'target_price_change', 'target_trend']
        feature_cols = [col for col in train_data.columns if col not in exclude_cols]
        
        if not feature_cols:
            raise ValueError("❌ No feature columns found for training")
        
        self.feature_columns = feature_cols
        X = train_data[feature_cols].copy()
        
        # Prepare trend targets
        if 'target_trend' in train_data.columns:
            y = train_data['target_trend'].astype(str)
        else:
            # Create trend categories from price_change
            price_change = train_data['target_price_change']
            trend = pd.cut(price_change, 
                         bins=[-np.inf, -0.02, 0.02, np.inf], 
                         labels=['down', 'stable', 'up'])
            y = trend.astype(str)
        
        # Remove rows with NaN values
        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        
        print(f"✅ Features prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"📊 Classes: {list(self.classes_)}")
        print(f"📊 Class distribution: {pd.Series(y).value_counts().to_dict()}")
        
        return X, pd.Series(y_encoded, index=X.index)
    
    def train(self, datasets: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, Any]:
        """
        🎯 Train KNN Classifier
        
        Args:
            datasets: Dict containing 'train' and 'test' dataframes
            **kwargs: Additional training parameters
            
        Returns:
            Dict with training metrics and results
        """
        print("🎯 Training KNN Classifier for trend prediction...")
        
        # Prepare data
        X_train, y_train = self._prepare_features(datasets)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        
        # Hyperparameter tuning
        if self.auto_tune:
            print("🔧 Tuning hyperparameters...")
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
            
            grid_search = GridSearchCV(
                KNeighborsClassifier(),
                param_grid,
                cv=min(5, len(X_train) // 10),  # Adaptive CV based on data size
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print(f"✅ Best parameters: {best_params}")
        else:
            self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
            self.model.fit(X_train_scaled, y_train)
            best_params = {'n_neighbors': self.n_neighbors}
        
        # Make predictions on training data
        y_pred_train = self.model.predict(X_train_scaled)
        
        # Calculate training metrics
        train_metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'best_params': best_params
        }
        
        # Cross-validation
        if len(X_train) > 100:
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train,
                cv=min(5, len(X_train) // 20),
                scoring='accuracy'
            )
            train_metrics['cv_accuracy_mean'] = cv_scores.mean()
            train_metrics['cv_accuracy_std'] = cv_scores.std()
        
        # Test on validation data if available
        if 'test' in datasets:
            test_metrics = self._evaluate_on_test(datasets['test'])
            train_metrics.update(test_metrics)
        
        # Update model state
        self.is_trained = True
        self.training_history = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'n_features': len(self.feature_columns),
            'n_samples': len(X_train),
            'n_classes': len(self.classes_),
            'metrics': train_metrics
        }
        
        # Update metadata
        self.metadata.update({
            'training_samples': len(X_train),
            'feature_count': len(self.feature_columns),
            'n_classes': len(self.classes_),
            'classes': list(self.classes_),
            'best_params': best_params,
            'last_trained': pd.Timestamp.now().isoformat()
        })
        
        # Print results
        print(f"✅ Training completed!")
        print(f"📊 Accuracy: {train_metrics['train_accuracy']:.4f}")
        if 'test_accuracy' in train_metrics:
            print(f"📊 Test Accuracy: {train_metrics['test_accuracy']:.4f}")
        
        return train_metrics
    
    def _evaluate_on_test(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model on test data"""
        # Prepare test features
        X_test = test_data[self.feature_columns].copy()
        
        # Prepare test targets
        if 'target_trend' in test_data.columns:
            y_test = test_data['target_trend'].astype(str)
        else:
            price_change = test_data['target_price_change']
            trend = pd.cut(price_change, 
                         bins=[-np.inf, -0.02, 0.02, np.inf], 
                         labels=['down', 'stable', 'up'])
            y_test = trend.astype(str)
        
        # Remove NaN values
        valid_idx = ~(X_test.isnull().any(axis=1) | y_test.isnull())
        X_test = X_test[valid_idx]
        y_test = y_test[valid_idx]
        
        # Encode labels (only transform, don't fit)
        try:
            y_test_encoded = self.label_encoder.transform(y_test)
        except ValueError:
            # Handle unseen labels
            valid_labels = y_test.isin(self.classes_)
            X_test = X_test[valid_labels]
            y_test = y_test[valid_labels]
            y_test_encoded = self.label_encoder.transform(y_test)
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predict
        y_pred_test = self.model.predict(X_test_scaled)
        
        return {
            'test_accuracy': accuracy_score(y_test_encoded, y_pred_test)
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        🔮 Make predictions
        
        Args:
            X: Feature dataframe
            
        Returns:
            Predicted class labels (decoded)
        """
        if not self.is_trained:
            raise ValueError("❌ Model must be trained before making predictions")
        
        # Ensure we have the right features
        if self.feature_columns:
            X = X[self.feature_columns].copy()
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict and decode labels
        y_pred_encoded = self.model.predict(X_scaled)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        🎲 Get prediction probabilities
        
        Args:
            X: Feature dataframe
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("❌ Model must be trained before making predictions")
        
        if self.feature_columns:
            X = X[self.feature_columns].copy()
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        📊 Evaluate model performance
        
        Args:
            X: Test features
            y: Test targets (string labels)
            
        Returns:
            Dict with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("❌ Model must be trained before evaluation")
        
        # Encode labels
        y_encoded = self.label_encoder.transform(y.astype(str))
        
        # Make predictions
        y_pred = self.predict(X)
        y_pred_encoded = self.label_encoder.transform(y_pred)
        
        return {
            'accuracy': accuracy_score(y_encoded, y_pred_encoded)
        }


class KNNRegressor(BaseModel):
    """
    📈 KNN Regressor for Crypto Price Prediction
    
    Features:
    - Price prediction
    - Hyperparameter tuning
    - Cross-validation
    - Feature scaling
    """
    
    def __init__(self, target_type: str = 'price', n_neighbors: int = 5, auto_tune: bool = True):
        """
        Initialize KNN Regressor
        
        Args:
            target_type: 'price' or 'price_change'
            n_neighbors: Number of neighbors (if auto_tune=False)
            auto_tune: Whether to automatically tune hyperparameters
        """
        super().__init__(
            model_name=f"knn_regressor_{target_type}",
            model_type="regression"
        )
        
        self.target_type = target_type
        self.n_neighbors = n_neighbors
        self.auto_tune = auto_tune
        self.scaler = StandardScaler()
        self.feature_columns = None
        
        # Update metadata
        self.metadata.update({
            'target_type': target_type,
            'n_neighbors': n_neighbors,
            'auto_tune': auto_tune,
            'algorithm': 'KNeighborsRegressor'
        })
    
    def _prepare_features(self, datasets: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        🔧 Prepare features and targets for regression
        
        Args:
            datasets: Dict containing train/test data
            
        Returns:
            Tuple of (features, targets)
        """
        print(f"🔧 Preparing features for {self.target_type} regression...")
        
        # Get training data
        train_data = datasets['train'].copy()
        
        # Define feature columns
        exclude_cols = ['date', 'symbol', 'target_price', 'target_price_change', 'target_trend']
        feature_cols = [col for col in train_data.columns if col not in exclude_cols]
        
        if not feature_cols:
            raise ValueError("❌ No feature columns found for training")
        
        self.feature_columns = feature_cols
        X = train_data[feature_cols].copy()
        
        # Prepare target
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
        🎯 Train KNN Regressor
        
        Args:
            datasets: Dict containing 'train' and 'test' dataframes
            **kwargs: Additional training parameters
            
        Returns:
            Dict with training metrics and results
        """
        print(f"🎯 Training KNN Regressor for {self.target_type}...")
        
        # Prepare data
        X_train, y_train = self._prepare_features(datasets)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        
        # Hyperparameter tuning
        if self.auto_tune:
            print("🔧 Tuning hyperparameters...")
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
            
            grid_search = GridSearchCV(
                KNeighborsRegressor(),
                param_grid,
                cv=min(5, len(X_train) // 10),
                scoring='r2',
                n_jobs=-1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print(f"✅ Best parameters: {best_params}")
        else:
            self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors)
            self.model.fit(X_train_scaled, y_train)
            best_params = {'n_neighbors': self.n_neighbors}
        
        # Make predictions on training data
        y_pred_train = self.model.predict(X_train_scaled)
        
        # Calculate training metrics
        train_metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'train_r2': r2_score(y_train, y_pred_train),
            'best_params': best_params
        }
        
        # Cross-validation
        if len(X_train) > 100:
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
            'best_params': best_params,
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
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predict
        y_pred_test = self.model.predict(X_test_scaled)
        
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
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
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