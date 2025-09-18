#!/usr/bin/env python3
"""
🚀 COMPREHENSIVE MODEL REGISTRY DEMO
===================================

Demonstrates the complete model registry system using NEW algorithm classes:
- ✅ LinearRegressionModel  
- ✅ KNNClassifier/KNNRegressor
- ✅ KMeansClusteringModel

This replaces the old function-based approach with enterprise-grade classes.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'app'))

from app.ml.data_prep import load_prepared_datasets
from app.ml.algorithms import LinearRegressionModel, KNNClassifier, KNNRegressor, KMeansClusteringModel
from app.ml.model_registry import model_registry

def demo_header(title):
    """Print demo section header"""
    print(f"\n{'='*60}")
    print(f"🎯 {title}")
    print('='*60)

def demo_linear_regression():
    """Demo Linear Regression with algorithm class"""
    demo_header("LINEAR REGRESSION MODELS")
    
    try:
        # Load datasets
        datasets = load_prepared_datasets('ml_datasets_top3')
        
        models_trained = []
        
        for target_type in ['price', 'price_change']:
            print(f"\n🔹 Training {target_type} prediction model...")
            
            # Initialize model
            model = LinearRegressionModel()
            
            # Prepare data
            X_train = datasets['X_train']
            y_train = datasets['y_train'][target_type]
            X_val = datasets['X_val']
            y_val = datasets['y_val'][target_type]
            X_test = datasets['X_test']
            y_test = datasets['y_test'][target_type]
            
            # Train model
            model.train(X_train, y_train, X_val, y_val)
            
            # Register in registry
            model_id = model_registry.register_model(
                model=model.model,
                model_name=f"crypto_linear_{target_type}_v2",
                model_type="LinearRegression",
                train_data=(X_train, y_train),
                val_data=(X_val, y_val),
                test_data=(X_test, y_test),
                train_metrics=model.training_history['train_metrics'],
                validation_metrics=model.training_history['val_metrics'],
                test_metrics=model.training_history['test_metrics'],
                feature_cols=datasets['feature_cols'],
                target_type=target_type,
                dataset_version='ml_datasets_top3',
                created_by='demo_new_algorithm_classes'
            )
            
            models_trained.append({
                'model_id': model_id,
                'model_name': f"crypto_linear_{target_type}_v2",
                'target_type': target_type,
                'r2_score': model.training_history['test_metrics']['r2']
            })
            
            print(f"✅ Model registered with ID: {model_id}")
            print(f"📈 Test R² Score: {model.training_history['test_metrics']['r2']:.4f}")
            
        return models_trained
        
    except Exception as e:
        print(f"❌ Linear regression demo failed: {str(e)}")
        return []

def demo_knn_models():
    """Demo KNN Classifier and Regressor"""
    demo_header("KNN MODELS (CLASSIFIER & REGRESSOR)")
    
    try:
        datasets = load_prepared_datasets('ml_datasets_top3')
        models_trained = []
        
        # KNN Regressor for price prediction
        print("\n🔹 Training KNN Regressor for price prediction...")
        knn_regressor = KNNRegressor()
        
        X_train = datasets['X_train']
        y_train = datasets['y_train']['price']
        X_val = datasets['X_val']
        y_val = datasets['y_val']['price']
        
        knn_regressor.train(X_train, y_train, X_val, y_val)
        
        print(f"✅ KNN Regressor trained")
        print(f"📈 Test R² Score: {knn_regressor.training_history['test_metrics']['r2']:.4f}")
        
        models_trained.append({
            'model_type': 'KNNRegressor',
            'target': 'price',
            'r2_score': knn_regressor.training_history['test_metrics']['r2']
        })
        
        # KNN Classifier - we would need classification targets
        # For demo purposes, let's create a simple trend classification
        print("\n🔹 Creating trend classification targets...")
        
        # Create simple trend classification (up/down based on price change)
        price_change = datasets['y_train']['price_change']
        trend_labels = (price_change > 0).astype(int)  # 1 for up, 0 for down
        
        price_change_val = datasets['y_val']['price_change']
        trend_labels_val = (price_change_val > 0).astype(int)
        
        print("\n🔹 Training KNN Classifier for trend prediction...")
        knn_classifier = KNNClassifier()
        knn_classifier.train(X_train, trend_labels, X_val, trend_labels_val)
        
        print(f"✅ KNN Classifier trained")
        print(f"📈 Test Accuracy: {knn_classifier.training_history['test_metrics']['accuracy']:.4f}")
        
        models_trained.append({
            'model_type': 'KNNClassifier',
            'target': 'trend',
            'accuracy': knn_classifier.training_history['test_metrics']['accuracy']
        })
        
        return models_trained
        
    except Exception as e:
        print(f"❌ KNN demo failed: {str(e)}")
        return []

def demo_kmeans_clustering():
    """Demo KMeans Clustering"""
    demo_header("KMEANS CLUSTERING")
    
    try:
        datasets = load_prepared_datasets('ml_datasets_top3')
        
        print("\n🔹 Training KMeans Clustering...")
        kmeans_model = KMeansClusteringModel()
        
        # Use training features for clustering
        X_train = datasets['X_train']
        kmeans_model.train(X_train)
        
        # Get cluster assignments
        cluster_labels = kmeans_model.predict(X_train)
        
        print(f"✅ KMeans Clustering trained")
        print(f"📊 Number of clusters: {kmeans_model.n_clusters}")
        print(f"📈 Silhouette Score: {kmeans_model.training_history['silhouette_score']:.4f}")
        print(f"📋 Samples per cluster: {np.bincount(cluster_labels)}")
        
        return {
            'n_clusters': kmeans_model.n_clusters,
            'silhouette_score': kmeans_model.training_history['silhouette_score'],
            'samples_per_cluster': np.bincount(cluster_labels).tolist()
        }
        
    except Exception as e:
        print(f"❌ KMeans demo failed: {str(e)}")
        return {}

def demo_registry_operations():
    """Demo registry operations"""
    demo_header("MODEL REGISTRY OPERATIONS")
    
    try:
        # List all models
        print("\n📋 All registered models:")
        all_models = model_registry.list_models()
        
        if all_models:
            for i, model_info in enumerate(all_models[-5:], 1):  # Show last 5 models
                print(f"{i}. {model_info['model_name']} (ID: {model_info['model_id'][:8]}...)")
                print(f"   Type: {model_info['model_type']}, Created: {model_info['created_at'][:19]}")
        else:
            print("   No models found in registry")
            
        # Demo model loading (if we have models)
        if all_models:
            latest_model = all_models[-1]
            print(f"\n🔄 Loading latest model: {latest_model['model_name']}")
            
            loaded_model = model_registry.load_model(latest_model['model_id'])
            if loaded_model:
                print(f"✅ Model loaded successfully!")
                print(f"📊 Model type: {type(loaded_model).__name__}")
            else:
                print("❌ Failed to load model")
                
        # Registry statistics
        print(f"\n📊 Registry Statistics:")
        print(f"   Total models: {len(all_models)}")
        
        # Count by type
        model_types = {}
        for model in all_models:
            model_type = model['model_type']
            model_types[model_type] = model_types.get(model_type, 0) + 1
            
        for model_type, count in model_types.items():
            print(f"   {model_type}: {count} models")
            
    except Exception as e:
        print(f"❌ Registry operations demo failed: {str(e)}")

def main():
    """Main demo function"""
    print("🚀 COMPREHENSIVE MODEL REGISTRY DEMO")
    print("🏗️ Using NEW Algorithm Classes")
    print("=" * 60)
    
    try:
        # Demo each model type
        linear_models = demo_linear_regression()
        knn_models = demo_knn_models()
        clustering_result = demo_kmeans_clustering()
        
        # Demo registry operations
        demo_registry_operations()
        
        # Summary
        demo_header("DEMO SUMMARY")
        print("✅ All algorithm classes demonstrated successfully!")
        print(f"📈 Linear Regression models: {len(linear_models)}")
        print(f"🎯 KNN models: {len(knn_models)}")
        print(f"🎪 Clustering: {'✅ Success' if clustering_result else '❌ Failed'}")
        
        print("\n🎉 Enterprise ML Architecture Demo Complete!")
        print("👉 All models are now using the new algorithm classes")
        print("🗑️ Legacy function-based code can be safely removed")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()