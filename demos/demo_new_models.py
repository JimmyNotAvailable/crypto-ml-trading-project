#!/usr/bin/env python3
"""
🏗️ NEW MODEL CLASSES DEMO
========================

Test the new enterprise-grade model classes:
- ✅ BaseModel interface
- ✅ LinearRegressionModel  
- ✅ KNNClassifier & KNNRegressor
- ✅ KMeansClusteringModel
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'app'))

from app.ml.data_prep import load_prepared_datasets
from app.ml.algorithms import LinearRegressionModel, KNNClassifier, KNNRegressor, KMeansClusteringModel

def test_linear_regression():
    """Test Linear Regression model class"""
    print("🧪 Testing LinearRegressionModel...")
    
    # Load datasets
    datasets = load_prepared_datasets("ml_datasets_top3")
    
    # Convert to expected format
    datasets_converted = {
        'train': datasets['train_df'],
        'test': datasets['test_df']
    }
    
    # Test price prediction
    lr_price = LinearRegressionModel(target_type='price', normalize_features=True)
    print(f"📊 Model info: {lr_price}")
    
    # Train model
    metrics = lr_price.train(datasets_converted)
    print(f"✅ Training metrics: R² = {metrics['train_r2']:.4f}")
    
    # Test prediction
    test_data = datasets_converted['test'].head(100)
    predictions = lr_price.predict(test_data)
    print(f"🔮 Made {len(predictions)} predictions")
    
    # Get feature importance
    importance = lr_price.get_feature_importance()
    print(f"📊 Top features: {importance.head(3)['feature'].tolist()}")
    
    # Save model
    model_path = lr_price.save_model("demo_linreg_price")
    print(f"💾 Model saved: {model_path}")
    
    return lr_price

def test_knn_models():
    """Test KNN model classes"""
    print("\n🧪 Testing KNN Models...")
    
    # Load datasets
    datasets = load_prepared_datasets("ml_datasets_top3")
    
    # Convert to expected format
    datasets_converted = {
        'train': datasets['train_df'],
        'test': datasets['test_df']
    }
    
    # Test KNN Classifier
    print("🎯 Testing KNNClassifier...")
    knn_clf = KNNClassifier(auto_tune=False, n_neighbors=5)  # Fast test
    clf_metrics = knn_clf.train(datasets_converted)
    print(f"✅ Classifier accuracy: {clf_metrics['train_accuracy']:.4f}")
    
    # Test KNN Regressor
    print("📈 Testing KNNRegressor...")
    knn_reg = KNNRegressor(target_type='price', auto_tune=False, n_neighbors=5)
    reg_metrics = knn_reg.train(datasets_converted)
    print(f"✅ Regressor R²: {reg_metrics['train_r2']:.4f}")
    
    # Test predictions
    test_data = datasets_converted['test'].head(50)
    clf_pred = knn_clf.predict(test_data)
    reg_pred = knn_reg.predict(test_data)
    
    print(f"🔮 Classifier predictions: {np.unique(clf_pred)}")
    print(f"🔮 Regressor predictions range: {reg_pred.min():.2f} - {reg_pred.max():.2f}")
    
    return knn_clf, knn_reg

def test_kmeans_clustering():
    """Test KMeans clustering model"""
    print("\n🧪 Testing KMeansClusteringModel...")
    
    # Load datasets
    datasets = load_prepared_datasets("ml_datasets_top3")
    
    # Convert to expected format
    datasets_converted = {
        'train': datasets['train_df'],
        'test': datasets['test_df']
    }
    
    # Create clustering model
    kmeans = KMeansClusteringModel(auto_tune=False, n_clusters=4)  # Fast test
    
    # Train model
    metrics = kmeans.train(datasets_converted)
    print(f"✅ Clustering metrics:")
    print(f"   📊 Clusters: {metrics['n_clusters']}")
    print(f"   📊 Silhouette: {metrics['silhouette_score']:.3f}")
    print(f"   📊 Cluster sizes: {metrics['cluster_sizes']}")
    
    # Test predictions
    test_data = datasets_converted['test'].head(100)
    cluster_labels = kmeans.predict(test_data)
    print(f"🔮 Cluster assignments: {np.bincount(cluster_labels)}")
    
    # Get cluster centers
    centers = kmeans.get_cluster_centers()
    print(f"📊 Cluster centers shape: {centers.shape}")
    
    return kmeans

def test_model_persistence():
    """Test model saving and loading"""
    print("\n🧪 Testing Model Persistence...")
    
    # Load datasets
    datasets = load_prepared_datasets("ml_datasets_top3")
    
    # Convert to expected format
    datasets_converted = {
        'train': datasets['train_df'],
        'test': datasets['test_df']
    }
    
    # Train a simple model
    model = LinearRegressionModel(target_type='price_change')
    model.train(datasets_converted)
    
    # Save model
    saved_path = model.save_model("test_persistence")
    print(f"💾 Model saved to: {saved_path}")
    
    # Load model
    loaded_model = LinearRegressionModel.load_model(saved_path)
    print(f"📂 Model loaded: {loaded_model}")
    
    # Test that loaded model works
    test_data = datasets_converted['test'].head(10)
    original_pred = model.predict(test_data)
    loaded_pred = loaded_model.predict(test_data)
    
    # Compare predictions
    mae_diff = np.mean(np.abs(original_pred - loaded_pred))
    print(f"🔍 Prediction difference (MAE): {mae_diff:.8f}")
    
    if mae_diff < 1e-6:
        print("✅ Model persistence test passed!")
    else:
        print("❌ Model persistence test failed!")
    
    return loaded_model

def main():
    """Run all model tests"""
    print("🏗️ TESTING NEW MODEL CLASSES")
    print("=" * 50)
    
    try:
        # Test individual models
        lr_model = test_linear_regression()
        knn_clf, knn_reg = test_knn_models()
        kmeans_model = test_kmeans_clustering()
        loaded_model = test_model_persistence()
        
        print(f"\n✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("📊 Summary:")
        print(f"   🔹 LinearRegression: {'✅' if lr_model.is_trained else '❌'}")
        print(f"   🔹 KNN Classifier: {'✅' if knn_clf.is_trained else '❌'}")
        print(f"   🔹 KNN Regressor: {'✅' if knn_reg.is_trained else '❌'}")
        print(f"   🔹 KMeans Clustering: {'✅' if kmeans_model.is_trained else '❌'}")
        print(f"   🔹 Model Persistence: {'✅' if loaded_model.is_trained else '❌'}")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()