#!/usr/bin/env python3
"""
🌳 RANDOM FOREST MODELS
====================

Enterprise Random Forest implementation với:
- 🎯 Regression & Classification support
- 📊 Feature importance analysis
- ⚡ Parallel training
- 🔧 Hyperparameter optimization
- 📈 Advanced performance metrics
"""

import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    classification_report, confusion_matrix, accuracy_score
)

from .base import BaseModel

class RandomForestModel(BaseModel):
    """
    🌳 Random Forest Model - Enterprise implementation
    
    Supports:
    - Regression tasks (price prediction)
    - Classification tasks (trend prediction)
    - Feature importance analysis
    - Hyperparameter optimization
    """
    
    def __init__(self, 
                 task_type: str = 'regression',
                 target_type: str = 'price',
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 **kwargs):
        """
        Initialize Random Forest Model
        
        Args:
            task_type: 'regression' hoặc 'classification'
            target_type: Type of target variable
            n_estimators: Số lượng trees
            max_depth: Độ sâu tối đa của trees
            min_samples_split: Số samples tối thiểu để split
            min_samples_leaf: Số samples tối thiểu ở leaf
            random_state: Random seed
            n_jobs: Số CPU cores (-1 = all cores)
        """
        model_name = f"random_forest_{task_type}_{target_type}"
        model_type = f"random_forest_{task_type}"
        super().__init__(model_name, model_type)
        
        self.task_type = task_type
        self.target_type = target_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Model initialization
        if task_type == 'regression':
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=n_jobs,
                **kwargs
            )
        elif task_type == 'classification':
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=n_jobs,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")
        
        # Metadata
        self.feature_names_ = None
        self.feature_importances_ = None
        self.training_metrics_ = None
        self.trained_ = False
    
    def _prepare_features(self, datasets: Dict[str, pd.DataFrame]) -> tuple:
        """
        🔧 Prepare features cho Random Forest training
        
        Args:
            datasets: Dict containing 'train' and 'test' dataframes
            
        Returns:
            Tuple of (X_train, y_train) processed data
        """
        print(f"🔧 Preparing features for {self.target_type} {self.task_type}...")
        
        train_df = datasets['train'].copy()
        
        # Feature columns (loại bỏ target columns)
        target_columns = ['target_price', 'target_price_change', 'target_trend', 'date', 'symbol']
        feature_cols = [col for col in train_df.columns if col not in target_columns]
        
        X_train = train_df[feature_cols]
        
        # Map target_type to actual column name
        target_column_map = {
            'price': 'target_price',
            'price_change': 'target_price_change',
            'trend': 'target_trend'
        }
        
        actual_target_col = target_column_map.get(self.target_type, self.target_type)
        
        if actual_target_col not in train_df.columns:
            raise ValueError(f"Target column '{actual_target_col}' not found in dataset. Available: {list(train_df.columns)}")
        
        y_train = train_df[actual_target_col]
        
        # Handle classification tasks
        if self.task_type == 'classification':
            # Convert continuous target to categories
            if self.target_type == 'price_change':
                # Classify price changes: up (1), down (-1), stable (0)
                y_train = pd.cut(y_train, bins=[-np.inf, -0.01, 0.01, np.inf], 
                               labels=[-1, 0, 1]).astype(int)
            elif self.target_type == 'price':
                # Classify price trends based on quantiles
                y_train = pd.qcut(y_train, q=3, labels=[0, 1, 2])
        
        # Remove missing values
        mask = ~(X_train.isnull().any(axis=1) | y_train.isnull())
        X_train = X_train[mask]
        y_train = y_train[mask]
        
        # Store feature names
        self.feature_names_ = list(X_train.columns)
        
        print(f"✅ Features prepared: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"📊 Target: {self.target_type} ({self.task_type})")
        
        if self.task_type == 'regression':
            print(f"📊 Target stats: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
        else:
            print(f"📊 Target distribution: {y_train.value_counts().to_dict()}")
        
        return X_train, y_train
    
    def train(self, datasets: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, Any]:
        """
        🌳 Train Random Forest model
        
        Args:
            datasets: Dict containing 'train' and 'test' dataframes
            **kwargs: Additional training parameters
            
        Returns:
            Dict with training metrics and results
        """
        print(f"🌳 Training Random Forest {self.task_type} for {self.target_type}...")
        
        # Prepare data
        X_train, y_train = self._prepare_features(datasets)
        
        # Training
        print(f"🚀 Training {self.n_estimators} trees...")
        start_time = datetime.now()
        
        self.model.fit(X_train, y_train)
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # Feature importances
        self.feature_importances_ = dict(zip(
            self.feature_names_, 
            self.model.feature_importances_
        ))
        
        # Predictions
        y_pred_train = self.model.predict(X_train)
        
        # Metrics
        if self.task_type == 'regression':
            metrics = self._calculate_regression_metrics(y_train, y_pred_train)
        else:
            metrics = self._calculate_classification_metrics(y_train, y_pred_train)
        
        # Test evaluation
        if 'test' in datasets:
            test_metrics = self._evaluate_test_set(datasets['test'])
            metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
        
        # Store training info
        self.training_metrics_ = metrics
        self.trained_ = True
        
        metrics.update({
            'training_time': training_time,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'task_type': self.task_type,
            'target_type': self.target_type
        })
        
        print(f"✅ Training completed in {training_time:.2f}s!")
        self._print_metrics(metrics)
        
        return metrics
    
    def _calculate_regression_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate regression metrics"""
        return {
            'r2_score': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def _calculate_classification_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate classification metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': self._calculate_f1_macro(y_true, y_pred)
        }
    
    def _calculate_f1_macro(self, y_true, y_pred) -> float:
        """Calculate macro F1 score"""
        from sklearn.metrics import f1_score
        return f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    def _evaluate_test_set(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model on test set"""
        # Prepare test data
        target_columns = ['target_price', 'target_price_change', 'target_trend', 'date', 'symbol']
        feature_cols = [col for col in test_df.columns if col not in target_columns]
        
        X_test = test_df[feature_cols]
        
        # Map target_type to actual column name
        target_column_map = {
            'price': 'target_price',
            'price_change': 'target_price_change',
            'trend': 'target_trend'
        }
        
        actual_target_col = target_column_map.get(self.target_type, self.target_type)
        y_test = test_df[actual_target_col]
        
        # Handle classification
        if self.task_type == 'classification':
            if self.target_type == 'price_change':
                y_test = pd.cut(y_test, bins=[-np.inf, -0.01, 0.01, np.inf], 
                              labels=[-1, 0, 1]).astype(int)
            elif self.target_type == 'price':
                y_test = pd.qcut(y_test, q=3, labels=[0, 1, 2])
        
        # Remove missing values
        mask = ~(X_test.isnull().any(axis=1) | y_test.isnull())
        X_test = X_test[mask]
        y_test = y_test[mask]
        
        # Predictions
        y_pred_test = self.model.predict(X_test)
        
        # Metrics
        if self.task_type == 'regression':
            return self._calculate_regression_metrics(y_test, y_pred_test)
        else:
            return self._calculate_classification_metrics(y_test, y_pred_test)
    
    def _print_metrics(self, metrics: Dict[str, Any]):
        """Print training metrics"""
        if self.task_type == 'regression':
            print(f"📊 R² Score: {metrics.get('r2_score', 0):.4f}")
            print(f"📊 RMSE: {metrics.get('rmse', 0):.4f}")
            print(f"📊 MAE: {metrics.get('mae', 0):.4f}")
            if 'test_r2_score' in metrics:
                print(f"📊 Test R²: {metrics['test_r2_score']:.4f}")
        else:
            print(f"📊 Accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"📊 F1 Macro: {metrics.get('f1_macro', 0):.4f}")
            if 'test_accuracy' in metrics:
                print(f"📊 Test Accuracy: {metrics['test_accuracy']:.4f}")
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        🎯 Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predictions array
        """
        if not self.trained_:
            raise ValueError("Model must be trained before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names_]
        
        return self.model.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        📊 Evaluate model performance
        
        Args:
            X: Test feature data
            y: Test target data
            
        Returns:
            Dict with evaluation metrics
        """
        if not self.trained_:
            raise ValueError("Model must be trained before evaluation")
        
        if y is None:
            raise ValueError("Target data y is required for evaluation")
        
        # Handle classification target transformation
        if self.task_type == 'classification':
            if self.target_type == 'price_change':
                y = pd.cut(y, bins=[-np.inf, -0.01, 0.01, np.inf], 
                          labels=[-1, 0, 1]).astype(int)
            elif self.target_type == 'price':
                y = pd.qcut(y, q=3, labels=[0, 1, 2])
        
        # Prepare features
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names_]
        
        # Predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        if self.task_type == 'regression':
            return self._calculate_regression_metrics(y, y_pred)
        else:
            return self._calculate_classification_metrics(y, y_pred)
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        📊 Get top N most important features
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dict of feature names and their importance scores
        """
        if not self.trained_:
            raise ValueError("Model must be trained first")
        
        # Sort by importance
        sorted_features = sorted(
            self.feature_importances_.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return dict(sorted_features[:top_n])
    
    def optimize_hyperparameters(self, 
                                datasets: Dict[str, pd.DataFrame],
                                param_grid: Optional[Dict] = None,
                                cv: int = 3,
                                scoring: str = None) -> Dict[str, Any]:
        """
        🔧 Optimize hyperparameters using GridSearchCV
        
        Args:
            datasets: Training datasets
            param_grid: Parameter grid for search
            cv: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Best parameters and scores
        """
        print(f"🔧 Optimizing Random Forest hyperparameters...")
        
        # Prepare data
        X_train, y_train = self._prepare_features(datasets)
        
        # Default parameter grid
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        # Default scoring
        if scoring is None:
            scoring = 'r2' if self.task_type == 'regression' else 'accuracy'
        
        # Grid search
        grid_search = GridSearchCV(
            self.model, 
            param_grid, 
            cv=cv, 
            scoring=scoring,
            n_jobs=self.n_jobs,
            verbose=1
        )
        
        print(f"🚀 Running grid search with {cv}-fold CV...")
        grid_search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        print(f"✅ Best score: {grid_search.best_score_:.4f}")
        print(f"🏆 Best params: {grid_search.best_params_}")
        
        return results
    
    def save_model(self, filename: str) -> str:
        """
        💾 Save trained model
        
        Args:
            filename: Model filename (without extension)
            
        Returns:
            Full path to saved model
        """
        if not self.trained_:
            raise ValueError("Model must be trained before saving")
        
        # Model registry integration
        from ..model_registry import register_model
        
        model_info = {
            'algorithm': 'random_forest',
            'task_type': self.task_type,
            'target_type': self.target_type,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'feature_names': self.feature_names_,
            'feature_importances': self.feature_importances_,
            'metrics': self.training_metrics_
        }
        
        # Register model
        model_id = f"rf_{self.task_type}_{self.target_type}"
        register_model(model_id, model_info)
        
        # Save model file
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent.parent
        models_dir = project_root / "models"
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / f"{filename}.joblib"
        
        joblib.dump({
            'model': self.model,
            'metadata': model_info
        }, model_path)
        
        print(f"✅ Model saved: {model_path}")
        return str(model_path)
    
    @classmethod
    def load_model(cls, model_path: str) -> 'RandomForestModel':
        """
        📁 Load trained model
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded RandomForestModel instance
        """
        data = joblib.load(model_path)
        model = data['model']
        metadata = data['metadata']
        
        # Recreate instance
        instance = cls(
            task_type=metadata['task_type'],
            target_type=metadata['target_type'],
            n_estimators=metadata['n_estimators'],
            max_depth=metadata['max_depth']
        )
        
        instance.model = model
        instance.feature_names_ = metadata['feature_names']
        instance.feature_importances_ = metadata['feature_importances']
        instance.training_metrics_ = metadata['metrics']
        instance.trained_ = True
        
        return instance