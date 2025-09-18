#!/usr/bin/env python3
"""
🚀 ENTERPRISE MODEL REGISTRY DEMONSTRATION - SIMPLIFIED
=====================================================

Simplified demo focusing on Linear Regression to showcase:
- ✅ Timestamp versioning
- ✅ Comprehensive metadata tracking  
- ✅ Production deployment workflow
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'app'))

from app.ml.data_prep import load_prepared_datasets
from app.ml.algorithms import LinearRegressionModel
from app.ml.model_registry import model_registry

def main():
    print("🚀 MODEL REGISTRY DEMONSTRATION")
    print("===============================")
    
    try:
        print("📊 Loading crypto datasets...")
        raw_datasets = load_prepared_datasets('ml_datasets_top3')
        
        # Convert to format expected by algorithm classes
        # Combine X and y data back into train/test DataFrames
        train_df = raw_datasets['X_train'].copy()
        for target in ['price', 'price_change']:
            train_df[target] = raw_datasets['y_train'][target]
        
        test_df = raw_datasets['X_test'].copy()
        for target in ['price', 'price_change']:
            test_df[target] = raw_datasets['y_test'][target]
        
        datasets = {
            'train': train_df,
            'test': test_df,
            'feature_cols': raw_datasets['feature_cols']
        }
        
        print("🤖 Training Linear Regression models...")
        
        models_trained = []
        
        for target_type in ['price', 'price_change']:
            print(f"\n🔹 Training {target_type} prediction model...")
            
            # Train model using algorithm class
            model = LinearRegressionModel(target_type=target_type)
            
            # Train model (datasets contains all needed data)
            metrics = model.train(datasets)
            
            # Register model in registry
            model_id = model_registry.register_model(
                model=model.model,
                model_name=f"crypto_linear_{target_type}",
                model_type="LinearRegression",
                train_data=(datasets['X_train'], datasets['y_train'][target_type]),
                val_data=(datasets['X_val'], datasets['y_val'][target_type]),
                test_data=(datasets['X_test'], datasets['y_test'][target_type]),
                train_metrics=model.training_history['train_metrics'],
                validation_metrics=model.training_history.get('val_metrics', {}),
                test_metrics=model.training_history.get('test_metrics', {}),
                feature_cols=datasets['feature_cols'],
                target_type=target_type,
                hyperparameters={},
                dataset_version="v2.1_crypto_hourly",
                created_by="Demo User"
            )
            
            models_trained.append({
                'model_id': model_id,
                'model_name': f"crypto_linear_{target_type}",
                'target_type': target_type,
                'r2_score': model.training_history.get('test_metrics', {}).get('r2', 0.0)
            })
            
            print(f"✅ Model registered with ID: {model_id}")
            test_r2 = model.training_history.get('test_metrics', {}).get('r2', 0.0)
            print(f"📈 Test R² Score: {test_r2:.4f}")
        
        print("\n🔍 Registry Contents:")
        print("=" * 50)
        
        models_df = model_registry.list_models()
        if not models_df.empty:
            for _, model in models_df.iterrows():
                print(f"🔸 {model['model_name']} v{model['version']}")
                print(f"   ID: {model['model_id']}")
                print(f"   Type: {model['model_type']}")
                print(f"   Test R²: {model['test_r2']:.4f}")
                print(f"   Created: {model['timestamp'][:19]}")
                print(f"   Status: {model['status']}")
                print()
        else:
            print("📭 No models found in registry")
        
        print(f"🎉 Successfully demonstrated model registry with {len(models_trained)} models!")
        print(f"💾 Registry location: {model_registry.registry_path}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()