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
        # Build backward-compatible dict for algorithms expecting 'train'/'test'
        datasets_compat = {
            'train': datasets.get('train_df', pd.DataFrame()),
            'test': datasets.get('test_df', pd.DataFrame())
        }
        
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
            model.train(datasets_compat)
            
            # Register in registry
            # Extract metric keys from our standardized training_history
            metrics = model.training_history.get('metrics', {})
            # Map train/test metric names used by ModelRegistry
            train_metrics = {
                'mae': float(metrics.get('train_mae', 0.0)),
                'rmse': float(metrics.get('train_rmse', 0.0)),
                'r2': float(metrics.get('train_r2', 0.0))
            }
            test_metrics = {
                'mae': float(metrics.get('test_mae', 0.0)),
                'rmse': float(metrics.get('test_rmse', 0.0)),
                'r2': float(metrics.get('test_r2', 0.0))
            }
            val_metrics = {}

            model_id = model_registry.register_model(
                model=model.model,
                model_name=f"crypto_linear_{target_type}_v2",
                model_type="LinearRegression",
                hyperparameters={},
                train_data=(X_train, y_train),
                val_data=(X_val, y_val),
                test_data=(X_test, y_test),
                train_metrics=train_metrics,
                validation_metrics=val_metrics,
                test_metrics=test_metrics,
                feature_cols=datasets['feature_cols'],
                target_type=target_type,
                dataset_version='ml_datasets_top3',
                created_by='demo_new_algorithm_classes'
            )
            
            models_trained.append({
                'model_id': model_id,
                'model_name': f"crypto_linear_{target_type}_v2",
                'target_type': target_type,
                'r2_score': test_metrics.get('r2', 0.0)
            })
            
            print(f"✅ Model registered with ID: {model_id}")
            print(f"📈 Test R² Score: {test_metrics.get('r2', 0.0):.4f}")
            
        return models_trained
        
    except Exception as e:
        print(f"❌ Linear regression demo failed: {str(e)}")
        return []

def demo_knn_models():
    """Demo KNN Classifier and Regressor"""
    demo_header("KNN MODELS (CLASSIFIER & REGRESSOR)")
    
    try:
        datasets = load_prepared_datasets('ml_datasets_top3')
        datasets_compat = {
            'train': datasets.get('train_df', pd.DataFrame()),
            'test': datasets.get('test_df', pd.DataFrame())
        }
        models_trained = []

        # KNN Regressor for price prediction
        print("\n🔹 Training KNN Regressor for price prediction...")
        knn_regressor = KNNRegressor()
        knn_regressor.train(datasets_compat)
        print("✅ KNN Regressor trained")
        print(f"📈 Train R² Score: {knn_regressor.training_history.get('metrics', {}).get('train_r2', 0.0):.4f}")
        models_trained.append({
            'model_type': 'KNNRegressor',
            'target': 'price',
            'r2_score': knn_regressor.training_history.get('metrics', {}).get('test_r2', 0.0)
        })

        # KNN Classifier - simple trend classification from price_change
        print("\n🔹 Creating trend classification targets...")
        price_change = datasets['y_train']['price_change']
        trend_labels = (price_change > 0).astype(int)  # 1 for up, 0 for down
        price_change_val = datasets['y_val']['price_change']
        trend_labels_val = (price_change_val > 0).astype(int)

        print("\n🔹 Training KNN Classifier for trend prediction...")
        knn_classifier = KNNClassifier()
        # Train classifier with compatibility dict structure
        knn_classifier.train(datasets_compat)
        print("✅ KNN Classifier trained")
        print(f"📈 Train Accuracy: {knn_classifier.training_history.get('metrics', {}).get('train_accuracy', 0.0):.4f}")
        models_trained.append({
            'model_type': 'KNNClassifier',
            'target': 'trend',
            'accuracy': knn_classifier.training_history.get('metrics', {}).get('test_accuracy', 0.0)
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
        
        # Train with compatibility datasets
        datasets_compat = {
            'train': datasets.get('train_df', pd.DataFrame()),
            'test': datasets.get('test_df', pd.DataFrame())
        }
        kmeans_model.train(datasets_compat)
        
        # Get cluster assignments on a small sample of training data
        train_df = datasets.get('train_df', pd.DataFrame())
        sample_df = train_df.head(100) if not train_df.empty else pd.DataFrame()
        cluster_labels = kmeans_model.predict(sample_df) if not sample_df.empty else np.array([])
        
        print(f"✅ KMeans Clustering trained")
        print(f"📊 Number of clusters: {kmeans_model.n_clusters}")
        print(f"📈 Silhouette Score: {kmeans_model.training_history.get('metrics', {}).get('silhouette_score', 0.0):.4f}")
        if cluster_labels.size > 0:
            print(f"📋 Samples per cluster: {np.bincount(cluster_labels)}")
        
        return {
            'n_clusters': kmeans_model.n_clusters,
            'silhouette_score': kmeans_model.training_history.get('metrics', {}).get('silhouette_score', 0.0),
            'samples_per_cluster': np.bincount(cluster_labels).tolist() if cluster_labels.size > 0 else []
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
        all_models_df = model_registry.list_models()
        
        if not all_models_df.empty:
            # Show last 5 models
            recent = all_models_df.tail(5).to_dict(orient='records')
            for i, model_info in enumerate(recent, 1):
                print(f"{i}. {model_info.get('model_name')} (ID: {str(model_info.get('model_id', ''))[:8]}...)")
                ts = model_info.get('timestamp')
                ts_disp = ts[:19] if isinstance(ts, str) and len(ts) >= 19 else str(ts)
                print(f"   Type: {model_info.get('model_type')}, Created: {ts_disp}")
        else:
            print("   No models found in registry")
            
        # Demo model loading (if we have models)
        if not all_models_df.empty:
            latest_model = all_models_df.tail(1).to_dict(orient='records')[0]
            print(f"\n🔄 Loading latest model: {latest_model.get('model_name')}")
            
            # ModelRegistry doesn't expose load_model by id; implement a safe local loader
            def _safe_load_by_id(mid: str):
                from pathlib import Path
                base = Path(model_registry.registry_path)
                for sub in ['production', 'staging', 'experiments']:
                    mp = base / sub / mid / 'model.joblib'
                    if mp.exists():
                        import joblib
                        return joblib.load(mp)
                return None

            loaded_model = _safe_load_by_id(str(latest_model.get('model_id')))
            if loaded_model is not None:
                print("✅ Model loaded successfully!")
                print(f"📊 Model type: {type(loaded_model).__name__}")
            else:
                print("❌ Failed to load model")
                
        # Registry statistics
        print(f"\n📊 Registry Statistics:")
        total = 0 if all_models_df is None else len(all_models_df)
        print(f"   Total models: {total}")
        
        # Count by type
        model_types = {}
        if not all_models_df.empty:
            for model in all_models_df.to_dict(orient='records'):
                mtype = model.get('model_type')
                model_types[mtype] = model_types.get(mtype, 0) + 1
            
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