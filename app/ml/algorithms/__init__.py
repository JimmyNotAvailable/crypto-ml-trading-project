"""
🏗️ ML ALGORITHMS PACKAGE
=======================

Enterprise-grade ML algorithm classes with standardized interfaces:
- ✅ BaseModel abstract interface
- ✅ LinearRegressionModel for price prediction
- ✅ KNNClassifier & KNNRegressor for pattern recognition
- ✅ KMeansClusteringModel for market segmentation
- ✅ RandomForestModel for ensemble learning
- ✅ Model versioning & metadata tracking
- ✅ Automatic model registry integration
"""

from .base import BaseModel
from .linear_regression import LinearRegressionModel
from .knn_models import KNNClassifier, KNNRegressor
from .clustering import KMeansClusteringModel
from .random_forest import RandomForestModel
from .logistic_regression import LogisticRegressionModel

__all__ = [
    'BaseModel',
    'LinearRegressionModel', 
    'KNNClassifier',
    'KNNRegressor',
    'KMeansClusteringModel',
    'RandomForestModel',
    'LogisticRegressionModel'
]